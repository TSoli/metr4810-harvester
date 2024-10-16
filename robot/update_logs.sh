#!/usr/bin/env bash

rm ./logs/*
rshell -p /dev/ttyACM0 cp /pyboard/logs/* ./logs/
rshell -p /dev/ttyACM0 rm /pyboard/logs/*
