#!/bin/bash

function press_to_continue() {
    read -n 1 -s -r -p $'\e[32mPress any key to continue\e[0m'
    echo ""
}
