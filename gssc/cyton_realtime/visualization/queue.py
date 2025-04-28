import queue
import threading
import logging

class VisualizationQueue:
    """Handles communication between background threads and main thread for visualization"""
    def __init__(self):
        self.queue = queue.Queue()
        self._processing = False
        self._processing_thread = None
        self.visualizer = None
        
    def start_processing(self, visualizer):
        """Start processing visualization updates in the main thread"""
        self.visualizer = visualizer
        self._processing = True
        self._processing_thread = threading.Thread(target=self._process_queue)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
    def stop_processing(self):
        """Stop processing visualization updates"""
        self._processing = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
            
    def _process_queue(self):
        """Process visualization updates from the queue"""
        while self._processing:
            try:
                # Get update from queue with timeout to allow checking _processing flag
                update = self.queue.get(timeout=0.1)
                if update is None:
                    continue
                    
                # Process the update
                epoch_data, sampling_rate, sleep_stage, time_offset, epoch_start_time = update
                self.visualizer._plot_polysomnograph(
                    epoch_data, sampling_rate, sleep_stage, time_offset, epoch_start_time
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"\rError processing visualization update: {str(e)}")
                
    def add_update(self, epoch_data, sampling_rate, sleep_stage, time_offset, epoch_start_time):
        """Add a visualization update to the queue"""
        self.queue.put((epoch_data, sampling_rate, sleep_stage, time_offset, epoch_start_time)) 