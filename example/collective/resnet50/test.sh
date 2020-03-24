#!/bin/bash
num_cards=8
while true ; do
  case "$1" in
    -cards) num_cards="$2" ; shift 2 ;;
    *)
       if [[ ${#1} > 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi
       ;;
  esac
done


if [[ $num_cards != 8 ]]; then
    visable_devices=""
    for (( t=0; t < $num_cards - 1; t++))  ; do
        visable_devices=$visable_devices$t","
    done
    visable_devices=$visable_devices$t
    export CUDA_VISIBLE_DEVICES=$visable_devices
fi
