#!/usr/bin/bash
set -ex

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --update-config-template)
      UPDATE_CONFIG_TEMPLATE=true
      shift
      ;;
    *)
      # unknown option
      shift
      ;;
  esac
done

rm -rf static

wget -O dist.tar.gz https://public-frontend-1300249583.cos.ap-nanjing.myqcloud.com/test-hp-metagpt-web/dist-20240417204906.tar.gz
tar xvzf dist.tar.gz
mv dist static
rm dist.tar.gz

if [ "$UPDATE_CONFIG_TEMPLATE" = true ]; then
    rm static/config.yaml
    ln -s ../config/template.yaml static/config.yaml
fi
