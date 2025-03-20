import asyncio
import datetime
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Coroutine, Optional

import aiofiles
from aiobotocore.session import get_session
from mdutils.mdutils import MdUtils
from metagpt.actions import Action
from metagpt.actions.action_output import ActionOutput
from metagpt.actions.design_api import WriteDesign
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.actions.project_management import WriteTasks
from metagpt.actions.summarize_code import SummarizeCode
from metagpt.actions.write_code import WriteCode
from metagpt.actions.write_prd import WritePRD
from metagpt.config import CONFIG
from metagpt.const import (
    COMPETITIVE_ANALYSIS_FILE_REPO,
    DATA_API_DESIGN_FILE_REPO,
    SEQ_FLOW_FILE_REPO,
    SERDESER_PATH,
)
from metagpt.roles import Architect, Engineer, ProductManager, ProjectManager, Role
from metagpt.schema import Message
from metagpt.team import Team
from metagpt.utils.common import any_to_str, read_json_file, write_json_file
from metagpt.utils.git_repository import GitRepository
from pydantic import BaseModel, Field
from zipstream import AioZipStream

_default_llm_stream_log = partial(print, end="")


class PackInfo(BaseModel):
    url: str


class RoleRun(Action):
    role: Role

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        action = self.role.rc.todo
        self.desc = f"{self.role.profile} {action.desc or str(action)}"


class PackProject(Action):
    role: Role

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.desc = "Pack the project with prd, design, code and more."

    async def run(self, key: str):
        url = await self.upload(key)
        info = PackInfo(url=url)
        mdfile = MdUtils(None)
        mdfile.new_line(mdfile.new_inline_link(url, url.rsplit("/", 1)[-1]))
        return ActionOutput(mdfile.get_md_text(), info)

    async def upload(self, key: str):
        files = []
        workspace = CONFIG.git_repo.workdir
        workspace = str(workspace)
        for r, _, fs in os.walk(workspace):
            _r = r[len(workspace) :].lstrip("/")
            for f in fs:
                files.append({"file": os.path.join(r, f), "name": os.path.join(_r, f)})
        # aiozipstream
        chunks = []
        async for chunk in AioZipStream(files, chunksize=32768).stream():
            chunks.append(chunk)
        return await get_download_url(b"".join(chunks), key)


class SoftwareCompany(Role):
    """封装软件公司成角色，以快速接入agent store。"""

    finish: bool = False
    company: Team = Field(default_factory=Team)
    active_role: Optional[Role] = None
    git_repo: Optional[GitRepository] = None
    max_auto_summarize_code: int = 0

    def __init__(self, use_code_review=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        engineer = Engineer(n_borg=5, use_code_review=use_code_review)
        self.company.hire([ProductManager(), Architect(), ProjectManager(), engineer])
        self._init_actions([PackProject(role=engineer)])

    def recv(self, message: Message) -> None:
        self.company.run_project(message.content)

    async def _think(self) -> Coroutine[Any, Any, bool]:
        """软件公司运行需要4轮

        BOSS            -> ProductManager -> Architect   -> ProjectManager -> Engineer
        BossRequirement -> WritePRD       -> WriteDesign -> WriteTasks     -> WriteCode ->
        """
        if self.finish:
            self.rc.todo = None
            return False

        if self.git_repo is not None:
            CONFIG.git_repo = self.git_repo

        environment = self.company.env
        for role in environment.roles.values():
            if await role._observe():
                await role._think()
                if isinstance(role.rc.todo, PrepareDocuments):
                    self.active_role = role
                    await self.act()
                    self.git_repo = CONFIG.git_repo
                    return await self._think()

                if isinstance(role.rc.todo, SummarizeCode):
                    return await self._think()

                self.rc.todo = RoleRun(role=role)
                self.active_role = role
                return True

        self._set_state(0)
        return True

    async def _act(self) -> Message:
        if self.git_repo is not None:
            CONFIG.git_repo = self.git_repo
            CONFIG.src_workspace = CONFIG.git_repo.workdir / CONFIG.git_repo.workdir.name
            CONFIG.max_auto_summarize_code = self.max_auto_summarize_code

        if isinstance(self.rc.todo, PackProject):
            workdir = CONFIG.git_repo.workdir
            name = workdir.name
            uid = workdir.parent.name
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            key = f"{uid}/metagpt-{name}-{now}.zip"
            output = await self.rc.todo.run(key)
            self.finish = True
            return Message(output.content, role=self.profile, cause_by=type(self.rc.todo))

        default_log_stream = CONFIG.get("LLM_STREAM_LOG", _default_llm_stream_log)

        start = False
        insert_code = False

        def log_stream(msg):
            nonlocal start, insert_code
            if not start:
                if msg.startswith("["):
                    msg = "```json\n" + msg
                    insert_code = True
                start = True
            return default_log_stream(msg)

        CONFIG.LLM_STREAM_LOG = log_stream

        output = await self.active_role._act()
        self.active_role._set_state(state=-1)
        self.active_role.publish_message(output)

        if insert_code:
            default_log_stream("\n```\n")

        cause_by = output.cause_by

        if cause_by == any_to_str(WritePRD):
            output = await self.format_prd(output)
        elif cause_by == any_to_str(WriteDesign):
            output = await self.format_system_design(output)
        elif cause_by == any_to_str(WriteTasks):
            output = await self.format_tasks(output)
        elif cause_by == any_to_str(WriteCode):
            output = await self.format_code(output)
        elif cause_by == any_to_str(SummarizeCode):
            output = await self.format_code_summary(output)
        return output

    async def format_prd(self, msg: Message):
        docs = [(k, v) for k, v in msg.instruct_content.docs.items()]
        prd_doc = docs[0][1]
        data = json.loads(prd_doc.content)

        mdfile = MdUtils(None)
        title = "Original Requirements"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_paragraph(data[title])

        title = "Product Goals"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_list(data[title], marked_with="1")

        title = "User Stories"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_list(data[title], marked_with="1")

        title = "Competitive Analysis"
        mdfile.new_header(2, title, add_table_of_contents=False)
        if all(i.count(":") == 1 for i in data[title]):
            mdfile.new_table(
                2, len(data[title]) + 1, ["Competitor", "Description", *(i for j in data[title] for i in j.split(":"))]
            )
        else:
            mdfile.new_list(data[title], marked_with="1")

        title = "Competitive Quadrant Chart"
        mdfile.new_header(2, title, add_table_of_contents=False)
        competitive_analysis_path = (
            CONFIG.git_repo.workdir / Path(COMPETITIVE_ANALYSIS_FILE_REPO) / Path(prd_doc.filename).with_suffix(".png")
        )

        if competitive_analysis_path.exists():
            key = str(competitive_analysis_path.relative_to(CONFIG.git_repo.workdir.parent.parent))
            url = await upload_file_to_s3(competitive_analysis_path, key)
            mdfile.new_line(mdfile.new_inline_image(title, url))
        else:
            mdfile.insert_code(data[title], "mermaid")

        title = "Requirement Analysis"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_paragraph(data[title])

        title = "Requirement Pool"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_table(
            2, len(data[title]) + 1, ["Task Description", "Priority", *(i for j in data[title] for i in j)]
        )

        title = "UI Design draft"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_paragraph(data[title])

        title = "Anything UNCLEAR"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_paragraph(data[title])
        content = mdfile.get_md_text()
        return Message(content, cause_by=msg.cause_by, role=msg.role)

    async def format_system_design(self, msg: Message):
        system_designs = [(k, v) for k, v in msg.instruct_content.docs.items()]
        system_design_doc = system_designs[0][1]
        data = json.loads(system_design_doc.content)

        mdfile = MdUtils(None)

        title = "Implementation approach"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_paragraph(data[title])

        title = "File list"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_list(data[title], marked_with="1")

        title = "Data structures and interfaces"
        mdfile.new_header(2, title, add_table_of_contents=False)

        data_api_design_path = (
            CONFIG.git_repo.workdir
            / Path(DATA_API_DESIGN_FILE_REPO)
            / Path(system_design_doc.filename).with_suffix(".png")
        )
        if data_api_design_path.exists():
            key = str(data_api_design_path.relative_to(CONFIG.git_repo.workdir.parent.parent))
            url = await upload_file_to_s3(data_api_design_path, key)
            mdfile.new_line(mdfile.new_inline_image(title, url))
        else:
            mdfile.insert_code(data[title], "mermaid")

        title = "Program call flow"
        mdfile.new_header(2, title, add_table_of_contents=False)
        seq_flow_path = (
            CONFIG.git_repo.workdir / SEQ_FLOW_FILE_REPO / Path(system_design_doc.filename).with_suffix(".png")
        )
        if seq_flow_path.exists():
            key = str(seq_flow_path.relative_to(CONFIG.git_repo.workdir.parent.parent))
            url = await upload_file_to_s3(seq_flow_path, key)
            mdfile.new_line(mdfile.new_inline_image(title, url))
        else:
            mdfile.insert_code(data[title], "mermaid")

        title = "Anything UNCLEAR"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_paragraph(data[title])
        content = mdfile.get_md_text()
        return Message(content, cause_by=msg.cause_by, role=msg.role)

    async def format_tasks(self, msg: Message):
        tasks = [(k, v) for k, v in msg.instruct_content.docs.items()]
        task_doc = tasks[0][1]
        data = json.loads(task_doc.content)

        mdfile = MdUtils(None)
        title = "Required Python packages"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.insert_code("\n".join(data[title]), "txt")

        title = "Required Other language third-party packages"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.insert_code("\n".join(data[title]), "txt")

        title = "Logic Analysis"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_table(
            2, len(data[title]) + 1, ["Filename", "Class/Function Name", *(i for j in data[title] for i in j)]
        )

        title = "Task list"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.new_list(data[title])

        title = "Full API spec"
        mdfile.new_header(2, title, add_table_of_contents=False)
        if data[title]:
            mdfile.insert_code(data[title], "json")

        title = "Shared Knowledge"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.insert_code(data[title], "python")

        title = "Anything UNCLEAR"
        mdfile.new_header(2, title, add_table_of_contents=False)
        mdfile.insert_code(data[title], "python")
        content = mdfile.get_md_text()
        return Message(content, cause_by=msg.cause_by, role=msg.role)

    async def format_code(self, msg: Message):
        data = msg.content.splitlines()
        workdir = CONFIG.git_repo.workdir
        code_root = workdir / workdir.name

        mdfile = MdUtils(None)

        for filename in data:
            mdfile.new_header(2, filename, add_table_of_contents=False)
            async with aiofiles.open(code_root / filename) as f:
                content = await f.read()
            suffix = filename.rsplit(".", maxsplit=1)[-1]
            mdfile.insert_code(content, "python" if suffix == "py" else suffix)
        return Message(mdfile.get_md_text(), cause_by=msg.cause_by, role=msg.role)

    async def format_code_summary(self, msg: Message):
        # TODO
        return msg

    async def think(self):
        await self._think()
        return self.rc.todo

    async def act(self):
        return await self._act()

    def serialize(self, stg_path: Path = None):
        stg_path = SERDESER_PATH.joinpath("software_company") if stg_path is None else stg_path

        team_info_path = stg_path.joinpath("software_company_info.json")
        write_json_file(team_info_path, self.model_dump(exclude={"company": True}))

        self.company.serialize(stg_path.joinpath("company"))  # save company alone

    @classmethod
    def deserialize(cls, stg_path: Path) -> "Team":
        """stg_path = ./storage/team"""
        # recover team_info
        software_company_info_path = stg_path.joinpath("software_company_info.json")
        if not software_company_info_path.exists():
            raise FileNotFoundError(
                "recover storage meta file `team_info.json` not exist, "
                "not to recover and please start a new project."
            )

        software_company_info: dict = read_json_file(software_company_info_path)

        # recover environment
        company = Team.deserialize(stg_path=stg_path.joinpath("company"))
        software_company_info.update({"company": company})

        return cls(**software_company_info)


async def upload_file_to_s3(filepath: str, key: str):
    async with aiofiles.open(filepath, "rb") as f:
        content = await f.read()
        return await get_download_url(content, key)


async def get_download_url(content: bytes, key: str) -> str:
    if CONFIG.get("STORAGE_TYPE") == "S3":
        session = get_session()
        async with session.create_client(
            "s3",
            aws_secret_access_key=CONFIG.get("S3_SECRET_KEY"),
            aws_access_key_id=CONFIG.get("S3_ACCESS_KEY"),
            endpoint_url=CONFIG.get("S3_ENDPOINT_URL"),
            use_ssl=CONFIG.get("S3_SECURE"),
        ) as client:
            # upload object to amazon s3
            bucket = CONFIG.get("S3_BUCKET")
            await client.put_object(Bucket=bucket, Key=key, Body=content)
            return f"{CONFIG.get('S3_ENDPOINT_URL')}/{bucket}/{key}"
    else:
        storage = CONFIG.get("LOCAL_ROOT", "storage")
        base_url = CONFIG.get("LOCAL_BASE_URL", "storage")
        filepath = Path(storage) / key
        filepath.parent.mkdir(exist_ok=True, parents=True)
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(content)
        return f"{base_url}/{key}"


async def main(idea, **kwargs):
    sc = SoftwareCompany(**kwargs)
    sc.recv(Message(idea))
    while await sc.think():
        print(await sc.act())


if __name__ == "__main__":
    asyncio.run(main())
