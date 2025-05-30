# parent.py
import time
import os
import multiprocessing
import pandas as pd
from datetime import datetime, timezone, timedelta
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
BoardShim.disable_board_logger()

def format_timestamp(ts):
    if ts is None:
        return "None"
    # Convert Unix timestamp to datetime in UTC
    utc_time = datetime.fromtimestamp(ts, timezone.utc)
    # Convert to Hawaii time (UTC-10)
    hawaii_time = utc_time - timedelta(hours=10)
    return hawaii_time.strftime('%Y-%m-%d %I:%M:%S %p HST')

def format_elapsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def create_trimmed_csv(input_file, output_file, skip_samples):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile):
            if idx >= skip_samples:
                outfile.write(line)


def run_board_stream(playback_file, conn):
    try:
        # Receive the start_first_data_ts from parent
        msg_type, start_first_data_ts = conn.recv()
        start_first_data_ts = float(start_first_data_ts) if start_first_data_ts is not None else None
        
        params = BrainFlowInputParams()
        params.board_id = BoardIds.PLAYBACK_FILE_BOARD
        params.file = playback_file
        params.master_board = BoardIds.CYTON_DAISY_BOARD
        # params.playback_file_max_count = 1
        # params.playback_speed = 1
        # params.playback_file_offset = start_offset

        board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
        board.prepare_session()
        board.config_board('old_timestamps')
        board.start_stream()

        time.sleep(0.1)  # 100ms pause to let the stream actually start

        timestamp_channel = BoardShim.get_timestamp_channel(params.master_board)
        last_valid_data_ts = None

        while True:
            data = board.get_board_data()

            if data.shape[1] == 0:
                print(f"\n[Child] Gap detected at {time.time()}, exiting and telling parent to restart. last_valid_data_ts: {last_valid_data_ts}")
                conn.send(('last_ts', last_valid_data_ts))
                print(f"[Child] sent last_valid_data_ts to parent at {time.time()}")
                conn.close()
                print(f"[Child] closed connection at {time.time()}")
                return  # 🔥 Exit the child normally

            timestamps = data[timestamp_channel]

            if start_first_data_ts is None:
                start_first_data_ts = float(timestamps[0])
                # Send the updated start_first_data_ts back to parent
                conn.send(('start_ts', start_first_data_ts))

            last_valid_data_ts = float(timestamps[-1])
            elapsed = last_valid_data_ts - start_first_data_ts
            print(f"\n[Child] Elapsed {format_elapsed_time(elapsed)} based on timestamps: {format_timestamp(start_first_data_ts)} - {format_timestamp(last_valid_data_ts)}")
            # log the amount of data received
            print(f"[Child] Amount of data received: {data.shape[1]}")
            # log the last timestamp
            print(f"[Child] Last timestamp: {last_valid_data_ts}")

            time.sleep(1)

    except Exception as e:
        print(f"[Child] Error: {e}")
        conn.close()

def main():
    playback_file = "data/realtime_inference_test/BrainFlow-RAW_2025-03-29_copy_moved_gap_earlier.csv"
    start_first_data_ts = None  # Keep this at module level for parent process

    if not os.path.isfile(playback_file):
        print(f"File not found: {playback_file}")
        return

    df = pd.read_csv(playback_file, sep='\t', header=None)
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)

    offset = 0  # Start from beginning

    while True:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=run_board_stream, args=(playback_file, child_conn))
        p.start()
        
        # Send the current start_first_data_ts to child
        parent_conn.send(('start_ts', start_first_data_ts))
        
        last_good_ts = None
        child_exited_normally = False
        
        # Wait for child to either send a message or exit
        while p.is_alive():
            if parent_conn.poll():
                msg_type, received = parent_conn.recv()
                if msg_type == 'start_ts':
                    start_first_data_ts = float(received) if received is not None else None
                    print(f"\n[Parent] Updated start_first_data_ts to: {start_first_data_ts}")
                elif msg_type == 'last_ts':
                    last_good_ts = float(received)
                    print(f"\n[Parent] Received last good timestamp: {last_good_ts}")
                    child_exited_normally = True
                    break
            time.sleep(0.1)
            
        # Clean up the child process - only terminate if still alive
        if p.is_alive():
            p.terminate()
        p.join()  # Always join to ensure proper cleanup
        
        # If child exited without sending a message, exit the program
        if not child_exited_normally:
            print("[Parent] Child exited without sending a message. Exiting program.")
            break

        if last_good_ts is None:
            print("[Parent] No valid timestamp received. Exiting.")
            break

        timestamps = df.iloc[:, timestamp_channel]
        next_rows = timestamps[timestamps > last_good_ts]

        if next_rows.empty:
            print("[Parent] No more data after last timestamp. Exiting.")
            break

        offset = int(next_rows.index[0])  # ✅ cast to real Python int
        print(f"[Parent] Restarting from new offset: {offset}")

        # Create trimmed CSV
        trimmed_file = f"data/offset_files/offset_{offset}_{os.path.basename(playback_file)}"
        create_trimmed_csv(playback_file, trimmed_file, offset)

        # Update playback file
        playback_file = trimmed_file
        print(f"\n[Parent] Updated playback file to: {playback_file}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()