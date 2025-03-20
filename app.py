#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import re
import shutil
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict

import fire
import tenacity
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from metagpt.config import CONFIG
from metagpt.logs import set_llm_stream_logfunc
from metagpt.schema import Message
from metagpt.utils.common import any_to_name, any_to_str
from openai import OpenAI
import mimetypes

from data_model import (
    LLMAPIkeyTest,
    MessageJsonModel,
    NewMsg,
    Sentence,
    Sentences,
    SentenceType,
    SentenceValue,
    ThinkActPrompt,
    ThinkActStep,
)
from message_enum import MessageStatus, QueryAnswerType
from software_company import RoleRun, SoftwareCompany

# 添加所有必要的 MIME 类型
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('text/html', '.html')

class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith('.js'):
            response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
        elif path.endswith('.css'):
            response.headers['Content-Type'] = 'text/css; charset=utf-8'
        return response

class Service:
    @classmethod
    async def create_message(cls, req_model: NewMsg, request: Request):
        """
        Session message stream
        """
        tc_id = 0
        task = None
        try:
            exclude_keys = CONFIG.get("SERVER_METAGPT_CONFIG_EXCLUDE", [])
            config = {k.upper(): v for k, v in req_model.config.items() if k not in exclude_keys}
            cls._set_context(config)
            msg_queue = deque()
            CONFIG.LLM_STREAM_LOG = lambda x: msg_queue.appendleft(x) if x else None

            role = SoftwareCompany()
            role.recv(message=Message(content=req_model.query))
            answer = MessageJsonModel(
                steps=[
                    Sentences(
                        contents=[
                            Sentence(
                                type=SentenceType.TEXT.value,
                                value=SentenceValue(answer=req_model.query),
                                is_finished=True,
                            ).model_dump()
                        ],
                        status=MessageStatus.COMPLETE.value,
                    )
                ],
                qa_type=QueryAnswerType.Answer.value,
            )

            async def stop_if_disconnect():
                while not await request.is_disconnected():
                    await asyncio.sleep(1)

                if task is None:
                    return

                if not task.done():
                    task.cancel()
                    logger.info(f"cancel task {task}")

            asyncio.create_task(stop_if_disconnect())

            while True:
                tc_id += 1
                if await request.is_disconnected():
                    return
                think_result: RoleRun = await role.think()
                if not think_result:  # End of conversion
                    break

                think_act_prompt = ThinkActPrompt(role=think_result.role.profile)
                think_act_prompt.update_think(tc_id, think_result)
                yield think_act_prompt.prompt + "\n\n"
                task = asyncio.create_task(role.act())

                while not await request.is_disconnected():
                    if msg_queue:
                        think_act_prompt.update_act(msg_queue.pop(), False)
                        yield think_act_prompt.prompt + "\n\n"
                        continue

                    if task.done():
                        break

                    await asyncio.sleep(0.5)
                else:
                    task.cancel()
                    return

                act_result = await task
                think_act_prompt.update_act(act_result)
                yield think_act_prompt.prompt + "\n\n"
                answer.add_think_act(think_act_prompt)
            yield answer.prompt + "\n\n"  # Notify the front-end that the message is complete.
        except asyncio.CancelledError:
            task.cancel()
        except tenacity.RetryError as retry_error:
            yield cls.handle_retry_error(tc_id, retry_error)
        except Exception as ex:
            description = str(ex)
            answer = traceback.format_exc()
            think_act_prompt = cls.create_error_think_act_prompt(tc_id, description, description, answer)
            yield think_act_prompt.prompt + "\n\n"
        finally:
            CONFIG.WORKSPACE_PATH: pathlib.Path
            if CONFIG.WORKSPACE_PATH.exists():
                shutil.rmtree(CONFIG.WORKSPACE_PATH)

    @staticmethod
    def create_error_think_act_prompt(tc_id: int, title, description: str, answer: str) -> ThinkActPrompt:
        step = ThinkActStep(
            id=tc_id,
            status="failed",
            title=title,
            description=description,
            content=Sentence(type=SentenceType.ERROR.value, id=1, value=SentenceValue(answer=answer), is_finished=True),
        )
        return ThinkActPrompt(step=step)

    @classmethod
    def handle_retry_error(cls, tc_id: int, error: tenacity.RetryError):
        # Known exception handling logic
        try:
            # Try to get the original exception
            original_exception = error.last_attempt.exception()
            while isinstance(original_exception, tenacity.RetryError):
                original_exception = original_exception.last_attempt.exception()

            name = any_to_str(original_exception)
            if re.match(r"^openai\.", name):
                return cls._handle_openai_error(tc_id, original_exception)

            if re.match(r"^httpx\.", name):
                return cls._handle_httpx_error(tc_id, original_exception)

            if re.match(r"^json\.", name):
                return cls._handle_json_error(tc_id, original_exception)

            return cls.handle_unexpected_error(tc_id, error)
        except Exception:
            return cls.handle_unexpected_error(tc_id, error)

    @classmethod
    def _handle_openai_error(cls, tc_id, original_exception):
        answer = original_exception.message
        title = f"OpenAI {any_to_name(original_exception)}"
        think_act_prompt = cls.create_error_think_act_prompt(tc_id, title, title, answer)
        return think_act_prompt.prompt + "\n\n"

    @classmethod
    def _handle_httpx_error(cls, tc_id, original_exception):
        answer = f"{original_exception}. {original_exception.request}"
        title = f"httpx {any_to_name(original_exception)}"
        think_act_prompt = cls.create_error_think_act_prompt(tc_id, title, title, answer)
        return think_act_prompt.prompt + "\n\n"

    @classmethod
    def _handle_json_error(cls, tc_id, original_exception):
        answer = str(original_exception)
        title = "MetaGPT Action Node Error"
        description = f"LLM response parse error. {any_to_str(original_exception)}: {original_exception}"
        think_act_prompt = cls.create_error_think_act_prompt(tc_id, title, description, answer)
        return think_act_prompt.prompt + "\n\n"

    @classmethod
    def handle_unexpected_error(cls, tc_id, error):
        description = str(error)
        answer = traceback.format_exc()
        think_act_prompt = cls.create_error_think_act_prompt(tc_id, description, description, answer)
        return think_act_prompt.prompt + "\n\n"

    @staticmethod
    def _set_context(context: Dict) -> Dict:
        uid = uuid.uuid4().hex
        context["WORKSPACE_PATH"] = pathlib.Path("workspace", uid)
        for old, new in (("DEPLOYMENT_ID", "DEPLOYMENT_NAME"), ("OPENAI_API_BASE", "OPENAI_BASE_URL")):
            if old in context and new not in context:
                context[new] = context[old]
        CONFIG.set_context(context)
        return context


default_llm_stream_log = partial(print, end="")


def llm_stream_log(msg):
    with contextlib.suppress():
        CONFIG._get("LLM_STREAM_LOG", default_llm_stream_log)(msg)


class ChatHandler:
    @staticmethod
    async def create_message(req_model: NewMsg, request: Request):
        """Message stream, using SSE."""
        event = Service.create_message(req_model, request)
        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(event, headers=headers, media_type="text/event-stream")


class LLMAPIHandler:
    @staticmethod
    async def check_openai_key(req_model: LLMAPIkeyTest):
        try:
            # Listing all available models.
            client = OpenAI(api_key=req_model.api_key)
            response = client.models.list()
            model_set = {model.id for model in response.data}
            if req_model.llm_type in model_set:
                logger.info("API Key is valid.")
                return JSONResponse({"valid": True})
            else:
                logger.info("API Key is invalid.")
                return JSONResponse({"valid": False, "message": "Model not found"})
        except Exception as e:
            # If the request fails, return False
            logger.info(f"Error: {e}")
            return JSONResponse({"valid": False, "message": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    loop.create_task(clear_storage())
    yield


app = FastAPI(lifespan=lifespan)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/storage",
    StaticFiles(directory="./storage/"),
    name="storage",
)

app.add_api_route(
    "/api/messages",
    endpoint=ChatHandler.create_message,
    methods=["post"],
    summary="Session message sending (streaming response)",
)
app.add_api_route(
    "/api/test-api-key",
    endpoint=LLMAPIHandler.check_openai_key,
    methods=["post"],
    summary="LLM APIkey detection",
)

# 使用自定义的静态文件处理
app.mount("/", CustomStaticFiles(directory="static", html=True), name="static")

set_llm_stream_logfunc(llm_stream_log)


def gen_file_modified_time(folder_path):
    yield os.path.getmtime(folder_path)
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            yield os.path.getmtime(file_path)


async def clear_storage(ttl: float = 1800):
    storage = pathlib.Path(CONFIG.get("LOCAL_ROOT", "storage"))
    logger.info("task `clear_storage` start running")

    while True:
        current_time = time.time()
        for i in os.listdir(storage):
            i = storage / i
            try:
                last_time = max(gen_file_modified_time(i))
                if current_time - last_time > ttl:
                    shutil.rmtree(i)
                    await asyncio.sleep(0)
                    logger.info(f"Deleted directory: {i}")
            except Exception:
                logger.exception(f"check {i} error")
        await asyncio.sleep(60)


def main():
    server_config = CONFIG.get("SERVER_UVICORN", {})
    uvicorn.run(app="__main__:app", **server_config)


if __name__ == "__main__":
    fire.Fire(main)
