#  Copyright (c) 2017-2018 Uber Technologies, Inc.
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
import logging
import sys
from time import sleep
from traceback import format_exc

import torch.multiprocessing as mp
from six.moves import queue

from petastorm.workers_pool import EmptyResultError, VentilatedItemProcessedMessage
# Defines how frequently will we check the stop event while waiting on a blocking queue
from petastorm.workers_pool.thread_pool import WorkerTerminationRequested

IO_TIMEOUT_INTERVAL_S = 0.001
# Amount of time we will wait on a the queue to get the next result. If no results received until then, we will
# recheck if no more items are expected to be ventilated
_VERIFY_END_OF_VENTILATION_PERIOD = 0.1

_CONTROL_FINISHED = ('_stop_event', 'FINISHED')


logger = logging.getLogger(__name__)


def run(worker_impl, ventilator_queue, results_queue, row_transform):
    while True:
        try:
            (args, kargs) = ventilator_queue.get(block=True, timeout=IO_TIMEOUT_INTERVAL_S)
            if not args and _CONTROL_FINISHED[0] in kargs and kargs[_CONTROL_FINISHED[0]] == _CONTROL_FINISHED[1]:
                # Check the stopping condition
                break
            worker_impl.process(*args, **kargs)
            worker_impl.publish_func(VentilatedItemProcessedMessage())
        except queue.Empty:
            pass
        except WorkerTerminationRequested:
            pass
        except Exception as e:  # pylint: disable=broad-except
            stderr_message = 'Worker %d terminated: unexpected exception:\n' % worker_impl.worker_id
            stderr_message += format_exc()
            sys.stderr.write(stderr_message)
            results_queue.put(e)
            break


class PytorchProcessPool(object):
    """This class has pool interface but performs all work in calls to get_results. It is sometimes convenient
    to substitute a real pool with this dummy implementation.

    Found this class useful when profiling worker code. When on a separate thread, the worker code was not observable
    (out of the box) by the profiler"""

    # Have workers argument just to make compatible with other pool implementations
    def __init__(self, workers_count=10, results_queue_size=50):
        """
        Constructor

        :param int workers_count: How many processes to use in the process pool
        :param int results_queue_size: How large to make the results queue
        """
        self._processes = []
        self._results_queue_size = results_queue_size
        self.workers_count = workers_count
        self._stopped = False

        self._ventilated_items = 0
        self._ventilated_items_processed = 0
        self._ventilator = None

        self._ventilator_queue = mp.Queue()
        self._results_queue = mp.Queue(self._results_queue_size)

    def start(self, worker_class, worker_args=None, ventilator=None):
        # Instantiate a single worker with all the args
        self._processes = []
        for worker_id in range(self.workers_count):
            worker_impl = worker_class(worker_id, self._stop_aware_put, worker_args)
            process = mp.Process(target=run,
                                 args=(worker_impl, self._ventilator_queue, self._results_queue, self._row_transform))
            self._processes.append(process)
            process.start()

        if ventilator:
            self._ventilator = ventilator
            self._ventilator.start()

    def ventilate(self, *args, **kargs):
        """Send a work item to a worker process."""
        self._ventilated_items += 1
        self._ventilator_queue.put((args, kargs))

    def get_results(self):
        """Returns results from worker pool or re-raise worker's exception if any happen in worker thread.

        :param timeout: If None, will block forever, otherwise will raise :class:`.TimeoutWaitingForResultError`
            exception if no data received within the timeout (in seconds)

        :return: arguments passed to ``publish_func(...)`` by a worker. If no more results are anticipated,
                 :class:`.EmptyResultError`.
        """

        while True:
            # If there is no more work to do, raise an EmptyResultError
            if self._results_queue.empty() and self._ventilated_items == self._ventilated_items_processed:
                # We also need to check if we are using a ventilator and if it is completed
                if not self._ventilator or self._ventilator.completed():
                    raise EmptyResultError()

            try:
                result = self._results_queue.get(timeout=_VERIFY_END_OF_VENTILATION_PERIOD)
                if isinstance(result, VentilatedItemProcessedMessage):
                    self._ventilated_items_processed += 1
                    if self._ventilator:
                        self._ventilator.processed_item()
                    continue
                elif isinstance(result, Exception):
                    self.stop()
                    self.join()
                    raise result
                else:
                    return result
            except queue.Empty:
                continue

    def stop(self):
        if self._ventilator:
            self._ventilator.stop()
        stop_event = {_CONTROL_FINISHED[0]: _CONTROL_FINISHED[1]}
        [self.ventilate(**stop_event) for _ in range(self.workers_count)]
        self._stopped = True

    def join(self):
        for process in self._processes:
            i = 0
            while process.is_alive() and i < 30:
                # Wait for process to stop
                i += 1
                sleep(.1)

            if process.is_alive():
                # Sometimes a process does not exit properly. In that case we just terminate it.
                process.terminate()
            else:
                process.join()

    @property
    def diagnostics(self):
        return dict()

    def _stop_aware_put(self, data):
        """This method is called to write the results to the results queue. We use ``put`` in a non-blocking way so we
        can gracefully terminate the worker thread without being stuck on :func:`Queue.put`.

        The method raises :class:`.WorkerTerminationRequested` exception that should be passed through all the way up to
        :func:`WorkerThread.run` which will gracefully terminate main worker loop."""
        while True:
            try:
                self._results_queue.put(data, block=True, timeout=IO_TIMEOUT_INTERVAL_S)
                return
            except queue.Full:
                pass

            if self._stopped:
                raise WorkerTerminationRequested()

