#!/usr/bin/env bash

pushd ~/Downloads/datasets
zip -r twitter_screenshots_localization_dataset.zip twitter_screenshots_localization_dataset -x '**/.DS_Store'
echo ""
unzip -l twitter_screenshots_localization_dataset.zip
popd
