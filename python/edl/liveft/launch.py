# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import signal

from edl.liveft.elastic import ElasticManager
from edl.liveft.elastic import LauncherInterface
from edl.liveft.elastic import ElasticStatus
from edl.liveft.elastic import ELASTIC_EXIT_CODE


def launch():
    # user interface for launching the pserver.
    # launch_ps()
    # return

    elastic = ElasticManager()

    signal.signal(signal.SIGTERM, elastic.signal_handler)
    signal.signal(signal.SIGABRT, elastic.signal_handler)
    signal.signal(signal.SIGINT, elastic.signal_handler)

    while True:

        # wait for all nodes ready to run
        elastic.wait()

        # run self with specified launcher
        elastic.run(LauncherInterface)

        # keep watching the health status of self and being notified for other's failure
        ret = elastic.watch()
        if ret == ElasticStatus.COMPLETED:
            break
        if ret == ElasticStatus.HOLD:
            continue
        if ret == ElasticStatus.EXIT:
            break
        if ret == ElasticStatus.ERROR:
            sys.exit(3)
        if ret == ElasticStatus.RESTART:
            sys.exit(ELASTIC_EXIT_CODE)

    if int(elastic.sigint) > 0:
        sys.exit(128 + int(elastic.sigint))
    else:
        sys.exit(0)


if __name__ == "__main__":
    launch()
