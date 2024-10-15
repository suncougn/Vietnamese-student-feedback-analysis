import pandas as pd
from multiprocessing import Process, Manager
from data_processor_pipeline.cleaner import *
import os

# Load the data
data = pd.read_csv('vietnamese_student_feedbacks.csv', encoding='utf-8-sig')

# Initialize preprocessor
preprocessor = data_clean()

# Function to clean a single text
def clean_one_text(idx, cleaned_list): 
    # Clean the text and store it in the shared list
    try:
	    cleaned_text = preprocessor.clean_text(data['sentence'].iloc[idx])
	    cleaned_list.append(cleaned_text)
	    return True
    except: 
        return False

# Function to clean text in parallel
def clean_text(idxs: list[int], cleaned_list):
    while idxs:  # Continue until idxs is empty
        try:
            idx = idxs.pop(0)  # Pop the first index
            result = clean_one_text(idx, cleaned_list)
        except IndexError:  # If list is empty, break the loop
            break
       	if result:
       		print(f'Done: {idx}/{len(data)}')
       	else:
       		print(f'Error: {idx}/{len(data)}')

if __name__ == "__main__":
    # List of indices for parallel processing
    indices = list(range(len(data)))
    
    # Number of processes to spawn
    n_proc = 8
    processes: list[Process] = []
    
    # Use Manager to share indices and cleaned data between processes
    with Manager() as manager:
        indices = manager.list(indices)  # Shared list of indices
        cleaned_list = manager.list()  # Shared list to store cleaned texts
        
        # Spawn processes
        for _ in range(n_proc):
            p = Process(target=clean_text, args=(indices, cleaned_list))
            processes.append(p)
        
        # Start processes
        for p in processes:
            p.start()
        
        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Save all cleaned texts to the CSV file once processing is complete
        pd.DataFrame({'cleaned': list(cleaned_list)}).to_csv('cleaned_feedbacks.csv', index=False, encoding='utf-8-sig')
