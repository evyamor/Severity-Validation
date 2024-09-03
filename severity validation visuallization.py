import tkinter as tk
from tkinter import scrolledtext
from tkinter import Tk, PhotoImage, Label
from PIL import Image, ImageTk
import sys
import threading
import queue
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import os
import subprocess

# Define colors palette for all graphs
color_palette = ['#c4bbbb', '#d6edb7', '#cf665b', '#f2f0f7', '#bcbddc',
                 '#fcfbfd', '#dadaeb', '#9e9ac8', '#ebc5dd',
                 '#c4b0d9', '#e5d8bd', '#d2e3e7', '#f5e6e8', '#e3d9e1',
                 '#f1eadf', '#f8d39b', '#eeb263', '#df941b',
                 '#c36000', '#b04800', '#84b7cf', '#5998b5', '#7a7dff', '#3773b3', '#5744fc', '#9fa14a']


class InteractiveLog:
    """
    Interactive log application
    Buttons : Run Program - initiates the entire main()
              Assess Thresholds - initiates assessment()
              Clear log - erase all content in the log application

    Elapsed time can be seen in the bottom left corner, it will also be printed upon termination
    """

    def __init__(self, master):
        self.master = master
        self.master.title("Severity Validation - Visualization")
        self.master.configure(bg="#f0f0f0")  # Set the background color of the title bar

        # Create a scrolled text widget with a light gray background
        self.text_widget = scrolledtext.ScrolledText(master, wrap=tk.WORD, bg="black", fg='white')
        self.text_widget.pack(expand=True, fill="both")

        # Redirect standard output to the text widget
        self.original_stdout = sys.stdout
        sys.stdout = self

        # # Handle background Geneyx logo
        # # Load the background image
        # image_path = "geneyxlogo.jpg"
        # try:
        #     # Load the image
        #     image = Image.open(image_path)
        #
        #     # Resize the image to fit the label
        #     desired_width = 25  # Adjust this value as needed
        #     desired_height = 25  # Adjust this value as needed
        #     image = image.resize((desired_width, desired_height), Image.ANTIALIAS)
        #
        #     # Convert the resized image to a Tkinter-compatible format
        #     background_image = ImageTk.PhotoImage(image)
        #
        #     # Create a label to hold the resized image
        #     self.background_label = tk.Label(master, image=background_image)
        #     self.background_label.place(x=0, y=350)  # Adjust the position as needed
        #
        # except IOError as e:
        #     print("Error loading image:", e)

        # Button to trigger the function with a teal background
        self.run_button = tk.Button(master, text="Run Program", command=self.run_function, bg="#008080", fg="white")
        self.run_button.pack()

        # Button to initiate another function (disabled initially)
        self.assessment_function_button = tk.Button(master, text="Assess Thresholds", command=self.assessment_function,
                                                    state=tk.DISABLED)
        self.assessment_function_button.pack()

        # Button to analyze unexpected variant file
        self.analyze_unexpected_function_button = tk.Button(master, text="Analyze Unexpected Variants",
                                                            command=self.analyze_unexpected_function,
                                                            state=tk.DISABLED)
        self.analyze_unexpected_function_button.pack()

        # Button to clear the log with a dark red background
        self.clear_button = tk.Button(master, text="Clear Log", command=self.clear_log, state=tk.DISABLED, bg="#8B0000",
                                      fg="white")
        self.clear_button.pack()

        # Button to export all print statements that were printed during the program execution
        self.export_button = tk.Button(master, text="Export", command=self.export, state=tk.DISABLED, bg="#25e658",
                                       fg="black")
        self.export_button.pack(anchor="e", padx=10)

        # Label to display elapsed time
        self.elapsed_time_label = tk.Label(master, text="Elapsed Time: 0s", bg="#f0f0f0", fg="black")
        self.elapsed_time_label.pack(anchor="w", padx=10)

        # Use a thread-safe queue to hold messages from the subprocess
        self.output_queue = queue.Queue()

        # Variable to store the time when button is clicked
        self.start_time = None
        self.prog_start_time = None

        # Write entry statement
        self.write("Welcome to Geneyx severity validation.\nPlease click on 'Run Program' button to initiate the program.\n"
                   "In this step, we organize and classify the data within the variants dataset, extracting valuable insights."
                   "We then present this information through visualizations, such as graphs, to provide a clear and informative representation of the dataset. \n"
                   "Once each step process is finished the next buttons will be pressable again. \n \n"
                   "Assess Thresholds button will verify the threshold table is correct and aligned with the QA system, if there are no differences\n"
                   "then the thresholds are identical. Else, there is a difference and the QA utilizes different thresholds, make sure this step indeed presents no differences.\n\n"
                   "Analyze Unexpected button will initiate a deeper analysis of the unexpected variants presenting more data for variants identification.\n\n"
                   "For exiting with all printed data exported to an output file click on the 'Export' button.\n\n")
        # Bind the closing event to a function
        self.master.protocol("WM_DELETE_WINDOW", self.stop_program)

    def disable_all_buttons(self):
        self.export_button["state"] = "disabled"
        self.run_button["state"] = "disabled"
        self.analyze_unexpected_function_button["state"] = "disabled"
        self.clear_button["state"] = "disabled"
        self.assessment_function_button["state"] = "disabled"


    def enable_all_buttons(self):
        self.assessment_function_button["state"] = "normal"
        self.analyze_unexpected_function_button["state"] = "normal"
        self.clear_button["state"] = "normal"
        self.export_button["state"] = "normal"
        self.run_button["state"] = "normal"

    def run_function(self):
        # Keep initial program starting time
        self.prog_start_time = time.time()
        # Record the start time when the button is clicked
        self.start_time = time.time()

        # Disable buttons
        self.disable_all_buttons()
        try:
            # Start a new thread to run the main function
            threading.Thread(target=self.run_main).start()
        except Exception as e:
            completion_message = f"An error occurred: {e}"
            self.write(completion_message)

    def run_main(self):
        try:
            main()
            completion_message = "Program Processing Execution completed successfully."
        except Exception as e:
            completion_message = f"An error occurred: {e}"

        # Print the completion message and elapsed time to the log app
        self.write(completion_message)
        elapsed_time = time.time() - self.start_time
        self.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

        # Enable buttons
        self.enable_all_buttons()
        # disable run program button
        self.run_button["state"] = "disabled"

    def assessment_function(self):
        # Create new starting time
        self.start_time = time.time()
        # Disable buttons while running
        self.disable_all_buttons()

        print("assessing thresholds")

        try:
            # Start a new thread to run the main function
            threading.Thread(target=self.run_assessment).start()
        except Exception as e:
            completion_message = f"An error occurred: {e}"
            self.write(completion_message)

    def run_assessment(self):
        try:
            assessment()

            completion_message = "Assessment Execution completed successfully."
        except Exception as e:
            completion_message = f"An error occurred: {e}"

        # Print the completion message and elapsed time to the log app
        self.write(completion_message)
        elapsed_time = time.time() - self.start_time
        self.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

        # Enable buttons
        self.enable_all_buttons()
        # disable assesment button & run program button
        self.assessment_function_button["state"] = "disabled"
        self.run_button["state"] = "disabled"

    def analyze_unexpected_function(self):
        # Create new starting time
        self.start_time = time.time()

        # Disable buttons while it's running
        self.disable_all_buttons()

        print("Analyzing unexpected variants")

        try:
            # Start a new thread to run the main function
            threading.Thread(target=self.run_analyze_unexpected).start()
        except Exception as e:
            completion_message = f"An error occurred: {e}"
            self.write(completion_message)

    def run_analyze_unexpected(self):
        try:
            analyze_unexpected()

            completion_message = "Unexpected Analysis Execution completed successfully."
        except Exception as e:
            completion_message = f"An error occurred: {e}"

        # Print the completion message and elapsed time to the log app
        self.write(completion_message)
        elapsed_time = time.time() - self.start_time
        self.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

        # Enable export button only post analysis finish
        self.export_button["state"] = "normal"

    def export(self):
        # Disable buttons while it's running
        self.disable_all_buttons()

        print("Exporting data to 'program export.txt' file, Program's process can be viewed step by step in this file.")
        try:
            completion_message = "Program Execution completed successfully."

            # Print the completion message and elapsed time to the log app
            self.write(completion_message)
            elapsed_time = time.time() - self.prog_start_time()
            self.write(f"Overall Elapsed Time: {elapsed_time:.2f} seconds")
            # Wait one second for all printing to be done before closing the application
            os.wait(1)
            # Start a new thread to run the main function
            threading.Thread(target=self.run_export).start()
        except Exception as e:
            completion_message = f"An error occurred: {e}"
            self.write(completion_message)

    def run_export(self):
        try:
            export_log_app(self)

            completion_message = "Execution completed successfully."
        except Exception as e:
            completion_message = f"An error occurred: {e}"

    def clear_log(self):
        # Clear the text widget
        self.text_widget.delete(1.0, tk.END)

    def write(self, text):
        # Write to the text widget and enqueue for updating
        self.output_queue.put(text)
        self.master.after(10, self.update_text_widget)  # Trigger the update

    def update_text_widget(self):
        # Update the Tkinter app with messages from the queue
        try:
            message = self.output_queue.get_nowait()
            self.text_widget.insert(tk.END, message + "\n")
            self.text_widget.see(tk.END)  # Scroll to the end
            self.master.after(10, self.update_text_widget)  # Continue updating

            # Update elapsed time label
            if self.start_time is not None:
                elapsed_time = time.time() - self.start_time
                self.elapsed_time_label.config(text=f"Elapsed Time: {elapsed_time:.2f} seconds")

        except queue.Empty:
            pass

    def stop_program(self):
        # Perform any necessary cleanup before closing the program
        # For example, save data, close files, etc.
        self.master.destroy()
        sys.exit()  # Exit the program


## POST QA ANALYSIS
# Parse and curate the results from .txt file to .txt file in TSV format, keep variants with ManeStatus = 0 or 1
def curate_result(input_file, output_file):
    """
    param input_file: SNV tsv file from the QA analysis
    param output_file: output tsv file path
    return: Curated file with variants tagged with ManeStatus = 0 or 1
    """
    # Print to log
    print(f'Start curating the file {input_file} for ManeStatus = 0 or ManeStatus = 1 \n')
    # Read TSV file into a Pandas DataFrame
    df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

    # Filter rows where ManeStatus is 0 or 1
    df_filtered = df[df['ManeStatus'].isin([0, 1])]

    # Write the filtered DataFrame to a new TSV file
    df_filtered.to_csv(output_file, sep='\t', index=False)

    print(f"Filtered data saved to: {output_file}")


# Find effect mapping according to effect found in QA
def mapping_filter_encodings_qa(file, output_file):
    """
    param file: Curated tsv variants file
    param output_file: output path
    return: All variants retagged with FILTERs according to VEP EFFECT
    """
    print(f'Start mapping FILTER encodings for all variants in the file: {file}\n')
    effects = {'LOF': 1, 'MISSENSE': 2, 'SPLICE': 3, 'STRTSTP': 4, 'OTHER': 5}
    print(f'Start mapping FILTER encodings for file: {file}\n')

    try:
        # Read the entire TSV file into a Pandas DataFrame
        df = pd.read_csv(file, sep='\t')

        # Check if 'FILTER' and 'EFFECT' columns are present
        if 'FILTER' not in df.columns or 'EFFECT' not in df.columns:
            print("Error: 'FILTER' or 'EFFECT' column not found in the file.")
            return

        # Apply the mapping function to the entire DataFrame
        df_result = process_chunk(df, effects)

        # Write the result to a TSV file
        df_result.to_csv(output_file, sep='\t', index=False)

        print(f'Finished mapping FILTER encodings. Saved to: {output_file}\n')

    except Exception as e:
        print(f"An error occurred: {e}")


# Function to process a chunk of the DataFrame, SUBFUNCTION for mapping_filter_encodings_qa
def process_chunk(chunk, effects):
    """
    param chunk: Dataframe
    param effects: Dictionary mapping for effects by filters
    return: Dataframe with updated FILTER column
    """

    def map_filter(row):
        current_filter = row['FILTER']
        new_filter = ''
        effect_value = 0

        # Haploinsufficiency
        new_filter = current_filter[0]

        # Effect
        effect_list = [word.strip() for word in row['EFFECT'].split(',')] if ',' in row['EFFECT'] else [
            str(row['EFFECT'])]
        effect = determine_effect_qa(effect_list)
        effect_value = effects.get(effect, 0)

        # Clinical Significance
        if any(num in current_filter for num in ['6', '7', '8', '9', '10']):
            effect_value += 5

        new_filter += str(effect_value)

        # Review Status
        new_filter += 'DS' if 'DS' in current_filter else 'SS'

        return new_filter

    # Apply the mapping function to the 'FILTER' column
    chunk['FILTER'] = chunk.apply(map_filter, axis=1)
    print(f'Finished processing a chunk')
    return chunk


# Remap filter encodings depending on QA effect, SUBFUNCTION for process_chunk
def determine_effect_qa(effects):
    """
    param effects: String containing effects for a variant
    return: Matched effect, i.e LOF, MISSENSE, SPLICE,START_STOP,OTHER
    """
    # Determines matching Effect group, LOF > MISSENSE > SPLICE > STRTSTP > OTHER
    effect_mapping = {
        "OTHER": ["FRAME_SHIFT_AFTER", "FRAME_SHIFT_BEFORE"],
        "LOF": ["FRAME_SHIFT", "STOP_GAINED", "SPLICE_SITE_DONOR", "SPLICE_SITE_ACCEPTOR"],
        "MISSENSE": ["NON_SYNONYMOUS_CODING"],
        # "SPLICE": ["SPLICE_SITE_REGION", "SYNONYMOUS_CODING"],
        "STRTSTP": ["START_GAINED", "START_LOST", "STOP_LOST"]
    }

    # Create a list to store matched categories
    matched_categories = []

    # Check each effect and append the matching category to the list
    for effect in effects:
        if pd.notna(effect):  # Skip nan values within the list
            for category, keywords in effect_mapping.items():
                if any(keyword.lower() in effect.lower() for keyword in keywords):
                    matched_categories.append(category)
    # Check for splice splice site region + non_synonymous_coding or synonymous_coding
    syn = 0
    spl = 0
    for effect in effects:
        # splice site region
        if "SPLICE_SITE_REGION" in effect:
            spl = 1
        # non_synonymous_coding or synonymous_coding
        if "SYNONYMOUS_CODING" in effect:
            syn = 1
    if syn == 1 and spl == 1:
        matched_categories.append("SPLICE")

    # Determine the final category based on priority, other first to remove frame_shit_after\before combinations
    priority_order = ["OTHER", "LOF", "SPLICE", "MISSENSE", "STRTSTP"]
    for category in priority_order:
        if category in matched_categories:
            return category

    # If none of the specific effects are found, return "OTHER"
    return "OTHER"


# Calculate each subgroup size in each file and export all sizes to one xlsx file named 'subgroups_sizes.xlsx'
def subgroups_length(file, outpath):
    """
    param files: List of variants files corresponding to haploinsufficiency group
    param outpath: Output file path
    return: Excel file subgroups_sizes.xlsx containing each subgroup(FILTER encoding) number of variants
    """
    print(f'Calculate number of variants corresponding to each subgroup type. i.e. FILTER type \n')
    # Create a dictionary to store subgroup sizes for each file
    subgroup_sizes_dict = {'File': [], 'FILTER': [], 'Subgroup Size': [], 'Interpretation': []}

    print(f'Started analysing subgroups lengths in the file: {file}   \n')
    # Read TSV file
    df = pd.read_csv(file, sep='\t')

    # Calculate subgroup sizes based on the 'FILTER' column
    subgroup_sizes = df['FILTER'].value_counts()

    # Append subgroup sizes to the dictionary
    for subgroup, size in subgroup_sizes.items():
        subgroup_sizes_dict['File'].append(file)
        subgroup_sizes_dict['FILTER'].append(subgroup)
        subgroup_sizes_dict['Subgroup Size'].append(size)

        # Use the filter_interp function to get the interpretation
        interpretation = filter_interp(subgroup)
        subgroup_sizes_dict['Interpretation'].append(interpretation)

    print(f'Finished analysing subgroups lengths in the: {file} file  \n')
    # Create a DataFrame from the dictionary
    result_df = pd.DataFrame(subgroup_sizes_dict)

    # Export the result to 'subgroups_sizes.xlsx'
    result_df.to_excel(outpath, index=False)
    print(f'Finished analysing all subgroups. Saved data can be found at {outpath}\n')


# interpretation of filter encoding, SUBFUNCTION of subgroups_length
def filter_interp(filter):
    """
     param filter: FILTER encoding, i.e, A1DS
     return: Definition of the encoding, the function decodes the filter and returns the matching definition.
             For instance, A1DS =  Haploinsufficiency 0, LOF, pathogenic, at least two stars
     """
    # Define interpretation of the subgroup
    subg = ''

    # Parse haploinsufficiency
    if 'A' in filter:
        subg += ' Haploinsufficiency 0, '
    elif 'B' in filter:
        subg += ' Haploinsufficiency 3, '
    elif 'C' in filter:
        subg += ' Haploinsufficiency 40, '

    # Parse effect group and clinical significance
    if '10' in filter:
        subg += 'OTHER, benign,  '
    elif '1' in filter:
        subg += 'LOF, pathogenic,  '
    elif '2' in filter:
        subg += 'MISSENSE, pathogenic,  '
    elif '3' in filter:
        subg += 'SPLICE, pathogenic,  '
    elif '4' in filter:
        subg += 'START\STOP, pathogenic,  '
    elif '5' in filter:
        subg += 'OTHER, pathogenic,  '
    elif '6' in filter:
        subg += 'LOF, benign,  '
    elif '7' in filter:
        subg += 'MISSENSE, benign,  '
    elif '8' in filter:
        subg += 'SPLICE, benign,  '
    else:
        if '9' in filter:
            subg += 'START\STOP, benign,  '

    # Parse review status
    if 'SS' in filter:
        subg += 'one star'
    elif 'DS' in filter:
        subg += 'at least two stars'

    return subg


# Calculate number of variants per gene for each hi class
def genes_lengths(files):
    """
        param files: List of variants files corresponding to haploinsufficiency group
        return: Excel file genes_lengths.xlsx containing each gene number of tested variants
        """
    # Create a dictionary to store gene lengths
    gene_lengths_dict = {'File': [], 'Gene Name': [], 'Number of Variants': []}

    for file in files:
        print(f'Started analyzing gene lengths in the {file} file\n')
        # Read the xlsx file
        df = pd.read_excel(file)

        # Group the DataFrame by 'GENEINFO' and count the number of variants for each gene
        gene_counts = df['GENEINFO'].value_counts().reset_index(name='Number of Variants')
        gene_counts.columns = ['Gene Name', 'Number of Variants']

        # Append gene lengths to the dictionary
        for index, row in gene_counts.iterrows():
            gene_lengths_dict['File'].append(file)
            gene_lengths_dict['Gene Name'].append(row['Gene Name'])
            gene_lengths_dict['Number of Variants'].append(row['Number of Variants'])

        print(f'Finished analyzing gene lengths in the {file} file\n')

    # Create a DataFrame from the dictionary
    result_df = pd.DataFrame(gene_lengths_dict)

    # Export the result to an Excel file named 'genes_lengths.xlsx'
    result_df.to_excel('genes_lengths.xlsx', index=False)
    print(f'Finished analyzing gene lengths in all files {files}\n')


# Divide the tsv file to 2 files, expected variants and unexpected variants According to the initial
# assumption; i,e. Pathogenic variants don't get Low severity, Benign variants don't get High severity
def expectancy_division(file, unexp_tsv_path, exp_tsv_path):
    """
    param file: TSV curated file path
    param unexp_tsv_path: unexpected variants file path
    param exp_tsv_path: expected variants file path
    return: creates the two unexpected and expected files according to
    the initial assumption ( Pathogenic + Low or Benign + High are unexpected )
    """
    # Print to log
    print(f'Started processing expectancy division in file: {file}')
    # Read TSV file
    df = pd.read_csv(file, sep='\t')

    # Define filter conditions for high severity
    high_severity_condition = (df['Severity'] == 'High') & (
        df['FILTER'].str.contains('|'.join(['6', '7', '8', '9', '10'])))

    low_severity_condition = (df['Severity'] == 'Low') & (
            df['FILTER'].str.contains('|'.join(['1', '2', '3', '4', '5'])) & ~df['FILTER'].str.contains('10'))

    # Filter unexpected and expected DataFrames
    unexpected_df = df[high_severity_condition | low_severity_condition]
    expected_df = df[~(high_severity_condition | low_severity_condition)]

    # Save DataFrames to TSV files
    unexpected_df.to_csv(unexp_tsv_path, sep='\t', index=False)
    expected_df.to_csv(exp_tsv_path, sep='\t', index=False)

    # Save unexpected dataframe to Excel
    unexp_excel_path = unexp_tsv_path.split('.')[0] + '.xlsx'
    unexpected_df.to_excel(unexp_excel_path)

    print(f"Unexpected variants saved to TSV: {unexp_tsv_path}")
    print(f"Unexpected variants were also saved in excel format for analysis under the same name")
    print(f"Expected variants saved to TSV: {exp_tsv_path}")


# Collect raw statistics data, Number of expected and unexpected, by categories , TP, TN, FP, FN
def collect_raw_data(unexp_path, exp_path):
    """
    param unexp_path: unexpected variants file path
    param exp_path: expected variants file path
    return: total_length - Total number of tested variants,
            exp_length - Number of expected variants,
            unexp_length - Number of unexpected variants,
            hi_exp_rate - Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per haploinsufficiency group,
            effect_exp_rate- Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per effect type,
            subgroup_exp_rate - Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per tested subgroup,
            roc - Dictionary of AUC-ROC values, TP = High + Pathogenic, TN = Low + Benign, FP = High + Benign, FN = Low + Pathogenic
    """
    # Print to log
    print(f'Starting collecting data on expectancy ratios from {unexp_path}, {exp_path}')
    # Define statistic values of interest
    # total number of variants tested
    total_length = 0
    # number of expected variants
    exp_length = 0
    # number of unexpected variants
    unexp_length = 0
    # Define TP, TN , FP , FN according to :
    # TP = High + Pathogenic, TN = Low + Benign, FP = High + Benign, FN = Low + Pathogenic
    roc = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    # The following tuples (EXPECTED, UNEXPECTED), rate calculated by (expected / (expected + unexpected ))*100
    # haploinsufficiency expectancy rate
    hi_exp_rate = {"Haploinsufficiency 0": (0, 0), "Haploinsufficiency 3": (0, 0), "Haploinsufficiency 40": (0, 0)}
    # effect expectancy rate
    effect_exp_rate = {"LOF": (0, 0), "MISSENSE": (0, 0), "SPLICE": (0, 0), "STARTSTOP": (0, 0), "OTHER": (0, 0)}
    clnsig_eff_pathogenic = {"LOF": (0, 0), "MISSENSE": (0, 0), "SPLICE": (0, 0), "STARTSTOP": (0, 0), "OTHER": (0, 0)}
    clnsig_eff_benign = {"LOF": (0, 0), "MISSENSE": (0, 0), "SPLICE": (0, 0), "STARTSTOP": (0, 0), "OTHER": (0, 0)}
    # Subgroup expectancy rate
    subgroup_exp_rate = {
        "A1SS": (0, 0),
        "A1DS": (0, 0),
        "A2SS": (0, 0),
        "A2DS": (0, 0),
        "A3SS": (0, 0),
        "A3DS": (0, 0),
        "A4SS": (0, 0),
        "A4DS": (0, 0),
        "A5SS": (0, 0),
        "A5DS": (0, 0),
        "A6SS": (0, 0),
        "A6DS": (0, 0),
        "A7SS": (0, 0),
        "A7DS": (0, 0),
        "A8SS": (0, 0),
        "A8DS": (0, 0),
        "A9SS": (0, 0),
        "A9DS": (0, 0),
        "A10SS": (0, 0),
        "A10DS": (0, 0),
        "B1SS": (0, 0),
        "B1DS": (0, 0),
        "B2SS": (0, 0),
        "B2DS": (0, 0),
        "B3SS": (0, 0),
        "B3DS": (0, 0),
        "B4SS": (0, 0),
        "B4DS": (0, 0),
        "B5SS": (0, 0),
        "B5DS": (0, 0),
        "B6SS": (0, 0),
        "B6DS": (0, 0),
        "B7SS": (0, 0),
        "B7DS": (0, 0),
        "B8SS": (0, 0),
        "B8DS": (0, 0),
        "B9SS": (0, 0),
        "B9DS": (0, 0),
        "B10SS": (0, 0),
        "B10DS": (0, 0),
        "C1SS": (0, 0),
        "C1DS": (0, 0),
        "C2SS": (0, 0),
        "C2DS": (0, 0),
        "C3SS": (0, 0),
        "C3DS": (0, 0),
        "C4SS": (0, 0),
        "C4DS": (0, 0),
        "C5SS": (0, 0),
        "C5DS": (0, 0),
        "C6SS": (0, 0),
        "C6DS": (0, 0),
        "C7SS": (0, 0),
        "C7DS": (0, 0),
        "C8SS": (0, 0),
        "C8DS": (0, 0),
        "C9SS": (0, 0),
        "C9DS": (0, 0),
        "C10SS": (0, 0),
        "C10DS": (0, 0),
    }

    # Read TSV file for unexpected variants
    unexp_df = pd.read_csv(unexp_path, sep='\t')

    for index, row in unexp_df.iterrows():
        # Check for haploinsufficiency and increment corresponding count
        current_filter = row['FILTER']
        if 'A' in current_filter:
            hi_exp_rate["Haploinsufficiency 0"] = (
                hi_exp_rate["Haploinsufficiency 0"][0], hi_exp_rate["Haploinsufficiency 0"][1] + 1)
        elif 'B' in current_filter:
            hi_exp_rate["Haploinsufficiency 3"] = (
                hi_exp_rate["Haploinsufficiency 3"][0], hi_exp_rate["Haploinsufficiency 3"][1] + 1)
        elif 'C' in current_filter:
            hi_exp_rate["Haploinsufficiency 40"] = (
                hi_exp_rate["Haploinsufficiency 40"][0], hi_exp_rate["Haploinsufficiency 40"][1] + 1)

        # Check for effect and increment corresponding count
        if '10' in current_filter or '5' in current_filter:
            effect_exp_rate["OTHER"] = (effect_exp_rate["OTHER"][0], effect_exp_rate["OTHER"][1] + 1)
            # clnsig_effect, Benign
            if '10' in current_filter:
                clnsig_eff_benign["OTHER"] = (clnsig_eff_benign["OTHER"][0], clnsig_eff_benign["OTHER"][1] + 1)
            else:
                # Pathogenic
                clnsig_eff_pathogenic["OTHER"] = (
                    clnsig_eff_pathogenic["OTHER"][0], clnsig_eff_pathogenic["OTHER"][1] + 1)
        elif '9' in current_filter or '4' in current_filter:
            effect_exp_rate["STARTSTOP"] = (effect_exp_rate["STARTSTOP"][0], effect_exp_rate["STARTSTOP"][1] + 1)
            # clnsig_effect, Benign
            if '9' in current_filter:
                clnsig_eff_benign["STARTSTOP"] = (
                    clnsig_eff_benign["STARTSTOP"][0], clnsig_eff_benign["STARTSTOP"][1] + 1)
            else:
                # Pathogenic
                clnsig_eff_pathogenic["STARTSTOP"] = (
                    clnsig_eff_pathogenic["STARTSTOP"][0], clnsig_eff_pathogenic["STARTSTOP"][1] + 1)
        elif '8' in current_filter or '3' in current_filter:
            effect_exp_rate["SPLICE"] = (effect_exp_rate["SPLICE"][0], effect_exp_rate["SPLICE"][1] + 1)
            # clnsig_effect, Benign
            if '8' in current_filter:
                clnsig_eff_benign["SPLICE"] = (
                    clnsig_eff_benign["SPLICE"][0], clnsig_eff_benign["SPLICE"][1] + 1)
            else:
                # Pathogenic
                clnsig_eff_pathogenic["SPLICE"] = (
                    clnsig_eff_pathogenic["SPLICE"][0], clnsig_eff_pathogenic["SPLICE"][1] + 1)
        elif '7' in current_filter or '2' in current_filter:
            effect_exp_rate["MISSENSE"] = (effect_exp_rate["MISSENSE"][0], effect_exp_rate["MISSENSE"][1] + 1)
            # clnsig_effect, Benign
            if '7' in current_filter:
                clnsig_eff_benign["MISSENSE"] = (clnsig_eff_benign["MISSENSE"][0], clnsig_eff_benign["MISSENSE"][1] + 1)
            else:
                # Pathogenic
                clnsig_eff_pathogenic["MISSENSE"] = (
                    clnsig_eff_pathogenic["MISSENSE"][0], clnsig_eff_pathogenic["MISSENSE"][1] + 1)
        elif '6' in current_filter or '1' in current_filter:
            effect_exp_rate["LOF"] = (effect_exp_rate["LOF"][0], effect_exp_rate["LOF"][1] + 1)
            # clnsig_effect, Benign
            if '6' in current_filter:
                clnsig_eff_benign["LOF"] = (clnsig_eff_benign["LOF"][0], clnsig_eff_benign["LOF"][1] + 1)
            else:
                # Pathogenic
                clnsig_eff_pathogenic["LOF"] = (
                    clnsig_eff_pathogenic["LOF"][0], clnsig_eff_pathogenic["LOF"][1] + 1)

        # Check for subgroup filter and increment corresponding count
        for subgroup in subgroup_exp_rate:
            if subgroup in current_filter:
                subgroup_exp_rate[subgroup] = (subgroup_exp_rate[subgroup][0], subgroup_exp_rate[subgroup][1] + 1)

        # Increase auROC stats if any number from 1 to 5 is present in current_filter
        # Pathogenic & Unexpected -> Pathogenic & Low -> FN
        if any(str(num) in current_filter for num in range(1, 6)):
            roc["fn"] += 1
        # Benign & Unexpected -> Benign & High -> FP
        elif any(str(num) in current_filter for num in range(6, 11)):
            roc["fp"] += 1

    unexp_length = unexp_df.shape[0]
    # Increase total length by number of unexpected variants
    total_length += unexp_length

    # Print to log
    print(f'Finished processing {unexp_path}\n')
    # Read TSV file for expected variants
    exp_df = pd.read_csv(exp_path, sep='\t')

    for index, row in exp_df.iterrows():
        # Check for haploinsufficiency and increment corresponding count
        current_filter = row['FILTER']
        if 'A' in current_filter:
            hi_exp_rate["Haploinsufficiency 0"] = (
                hi_exp_rate["Haploinsufficiency 0"][0] + 1, hi_exp_rate["Haploinsufficiency 0"][1])
        elif 'B' in current_filter:
            hi_exp_rate["Haploinsufficiency 3"] = (
                hi_exp_rate["Haploinsufficiency 3"][0] + 1, hi_exp_rate["Haploinsufficiency 3"][1])
        elif 'C' in current_filter:
            hi_exp_rate["Haploinsufficiency 40"] = (
                hi_exp_rate["Haploinsufficiency 40"][0] + 1, hi_exp_rate["Haploinsufficiency 40"][1])

        # Check for effect and increment corresponding count
        if '10' in current_filter or '5' in current_filter:
            effect_exp_rate["OTHER"] = (effect_exp_rate["OTHER"][0] + 1, effect_exp_rate["OTHER"][1])
            # clnsig_effect, Benign
            if '10' in current_filter:
                clnsig_eff_benign["OTHER"] = (clnsig_eff_benign["OTHER"][0] + 1, clnsig_eff_benign["OTHER"][1])
            else:
                # Pathogenic
                clnsig_eff_pathogenic["OTHER"] = (
                    clnsig_eff_pathogenic["OTHER"][0] + 1, clnsig_eff_pathogenic["OTHER"][1])
        elif '9' in current_filter or '4' in current_filter:
            effect_exp_rate["STARTSTOP"] = (effect_exp_rate["STARTSTOP"][0] + 1, effect_exp_rate["STARTSTOP"][1])
            # clnsig_effect, Benign
            if '9' in current_filter:
                clnsig_eff_benign["STARTSTOP"] = (
                    clnsig_eff_benign["STARTSTOP"][0] + 1, clnsig_eff_benign["STARTSTOP"][1])
            else:
                # Pathogenic
                clnsig_eff_pathogenic["STARTSTOP"] = (
                    clnsig_eff_pathogenic["STARTSTOP"][0] + 1, clnsig_eff_pathogenic["STARTSTOP"][1])
        elif '8' in current_filter or '3' in current_filter:
            effect_exp_rate["SPLICE"] = (effect_exp_rate["SPLICE"][0] + 1, effect_exp_rate["SPLICE"][1])
            # clnsig_effect, Benign
            if '8' in current_filter:
                clnsig_eff_benign["SPLICE"] = (
                    clnsig_eff_benign["SPLICE"][0] + 1, clnsig_eff_benign["SPLICE"][1])
            else:
                # Pathogenic
                clnsig_eff_pathogenic["SPLICE"] = (
                    clnsig_eff_pathogenic["SPLICE"][0] + 1, clnsig_eff_pathogenic["SPLICE"][1])
        elif '7' in current_filter or '2' in current_filter:
            effect_exp_rate["MISSENSE"] = (effect_exp_rate["MISSENSE"][0] + 1, effect_exp_rate["MISSENSE"][1])
            # clnsig_effect, Benign
            if '7' in current_filter:
                clnsig_eff_benign["MISSENSE"] = (clnsig_eff_benign["MISSENSE"][0] + 1, clnsig_eff_benign["MISSENSE"][1])
            else:
                # Pathogenic
                clnsig_eff_pathogenic["MISSENSE"] = (
                    clnsig_eff_pathogenic["MISSENSE"][0] + 1, clnsig_eff_pathogenic["MISSENSE"][1])
        elif '6' in current_filter or '1' in current_filter:
            effect_exp_rate["LOF"] = (effect_exp_rate["LOF"][0] + 1, effect_exp_rate["LOF"][1])
            # clnsig_effect, Benign
            if '6' in current_filter:
                clnsig_eff_benign["LOF"] = (clnsig_eff_benign["LOF"][0] + 1, clnsig_eff_benign["LOF"][1])
            else:
                # Pathogenic
                clnsig_eff_pathogenic["LOF"] = (
                    clnsig_eff_pathogenic["LOF"][0] + 1, clnsig_eff_pathogenic["LOF"][1])

        # Increment corresponding count  subgroup filter
        subgroup_exp_rate[current_filter] = (
            subgroup_exp_rate[current_filter][0] + 1, subgroup_exp_rate[current_filter][1])

        # Increase auROC stats
        # Pathogenic & Expected -> Pathogenic & High -> TP
        if any(str(num) in current_filter for num in range(1, 6)):
            roc["tp"] += 1
        # Benign & Expected -> Benign & Low -> TN
        elif any(str(num) in current_filter for num in range(6, 11)):
            roc["tn"] += 1

    exp_length = exp_df.shape[0]
    # Increase total length by number of expected variants
    total_length += exp_length
    print(f'Finished processing {exp_path}\n')
    # Calculate and print rates only if there are variants
    if total_length > 0:
        print(
            f'Total number of variants, {total_length}, Number of expected variants {exp_length}, Number of unexpected variants {unexp_length} ')
        # Print haploinsufficiency expectancy rates
        for hi_category, (expected, unexpected) in hi_exp_rate.items():
            rate = (expected / (expected + unexpected)) * 100 if (expected + unexpected) != 0 else 0
            print(
                f'For Haploinsufficiency category, {hi_category}, Number of expected variants {expected}, Number of unexpected variants {unexpected} ')
            print(f"{hi_category} expectancy rate: {rate} %")

        # Print effect expectancy rates
        for effect_category, (expected, unexpected) in effect_exp_rate.items():
            rate = (expected / (expected + unexpected)) * 100 if (expected + unexpected) != 0 else 0
            print(
                f'For effect category, {effect_category}, Number of expected variants {expected}, Number of unexpected variants {unexpected} ')
            print(f"{effect_category} expectancy rate: {rate} %")

        # Print subgroup expectancy rates
        for subgroup, (expected, unexpected) in subgroup_exp_rate.items():
            if (expected + unexpected) > 0:  # Skip printing if no variants were tested
                rate = (expected / (expected + unexpected)) * 100 if (expected + unexpected) != 0 else 0
                print(
                    f'For subgroup category, {subgroup}, Number of expected variants {expected}, Number of unexpected variants {unexpected} ')
                print(f"{subgroup} subgroup expectancy rate: {rate} %")

    # Return calculated rates
    return total_length, exp_length, unexp_length, hi_exp_rate, effect_exp_rate, clnsig_eff_benign, clnsig_eff_pathogenic, subgroup_exp_rate, roc


# Visualize the data
def visualize_data(total_length, expected_length, unexpected_length, hi_statistics, effect_statistics,
                   clnsig_effect_benign, clnsig_effect_pathogenic, filters_statistics, auroc, file_path):
    """
    param total_length: Total number of tested variants
    param expected_length: Number of expected variants
    param unexpected_length: Number of unexpected variants
    param hi_statistics: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per haploinsufficiency group
    param effect_statistics: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per effect type
    param filters_statistics: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per tested subgroup
    param auroc: Dictionary of AUC-ROC values, TP = High + Pathogenic, TN = Low + Benign, FP = High + Benign, FN = Low + Pathogenic
    param file_path: input file path for extracting severity division statistics
    return: Creates graphs for relevant scores
    """
    subgroup_expectancy_graph(filters_statistics, 'subgroups_expectancy_graph.png')
    effect_expectancy_graph(effect_statistics, 'effect_expectancy_graph.png')
    clnsig_effect_expectancy_graph(clnsig_effect_benign, clnsig_effect_pathogenic, 'clnsig_effect_graph.png')
    hi_expectancy_graph(hi_statistics, 'hi_expectancy_graph.png')
    auroc_dot(auroc, 'auroc_graph.png')
    severity_pie_chart(file_path, 'severity_distribution.png')
    overall_success_pie(total_length, expected_length, unexpected_length, 'Total_success_ratio.png')
    clnsig_pie_chart(file_path, 'clinical_significance_distribution.png')
    sunburst_data(filters_statistics, 'data_sunburst.html')


def sunburst_filter_decode(filter):
    """
    param filter: FILTER, i.e A1SS
    return:  FILTER decoded
    """
    # Define interpretation of the subgroup
    haplo = ''
    effect = ''
    rev = ''
    clinsig = ''

    # Parse haploinsufficiency
    if 'A' in filter:
        haplo = ' Haplo 0 '
    elif 'B' in filter:
        haplo = ' Haplo 3 '
    elif 'C' in filter:
        haplo = ' Haplo 40 '
    # Parse effect group and clinical significance
    if '10' in filter:
        effect = 'OTHER'
        clinsig = 'BEN'
    elif '1' in filter:
        effect = 'LOF'
        clinsig = 'PAT'
    elif '2' in filter:
        effect = 'MISSENSE'
        clinsig = 'PAT'
    elif '3' in filter:
        effect = 'SPLICE'
        clinsig = 'PAT'
    elif '4' in filter:
        effect = 'Start_Stop'
        clinsig = 'PAT'
    elif '5' in filter:
        effect = 'OTHER'
        clinsig = 'PAT'
    elif '6' in filter:
        effect = 'LOF'
        clinsig = 'BEN'
    elif '7' in filter:
        effect = 'MISSENSE'
        clinsig = 'BEN'
    elif '8' in filter:
        effect = 'SPLICE'
        clinsig = 'BEN'
    else:
        if '9' in filter:
            effect = 'Start_Stop'
            clinsig = 'BEN'
    # Parse review status
    if 'SS' in filter:
        rev = '1 star'
    elif 'DS' in filter:
        rev = '2 stars +'
    return haplo, effect, rev, clinsig


def sunburst_data(stats, save_path):
    all_data = []
    for filter, tuple_rate in stats.items():
        data = {"Haplo": '', "Effect": '', "Rev": '', "ClinSig": '', "Exp": 0,
                "Unexp": 0}
        data["Haplo"], data["Effect"], data["Rev"], data["ClinSig"] = sunburst_filter_decode(
            filter)
        data["Exp"] = tuple_rate[0]
        data["Unexp"] = tuple_rate[1]
        all_data.append(data)

    sb_df = pd.DataFrame(all_data)

    # Reshape DataFrame from wide to long format
    melted_df = pd.melt(sb_df,
                        id_vars=['Haplo', 'Effect', 'Rev', 'ClinSig'],
                        value_vars=['Exp', 'Unexp'],
                        var_name='Count_Type',
                        value_name='Count')

    # Create the sunburst chart using Plotly Express
    fig = px.sunburst(melted_df,
                      path=['Haplo', 'Effect', 'Rev', 'ClinSig', 'Count_Type'],
                      values='Count',
                      color='Count_Type',  # Use the 'Color' column for coloring
                      color_discrete_map={'(?)': color_palette[0], 'Exp': color_palette[1], 'Unexp': color_palette[2]},
                      title="Severity Validation Subgroups Sunburst Chart")

    # Add percentage and count labels to each slice
    fig.update_traces(textinfo='label+percent entry+value', textfont_size=12)

    unique_haplo = melted_df['Haplo'].unique()
    unique_effect = melted_df['Effect'].unique()
    unique_rev = melted_df['Rev'].unique()
    unique_clinsig = melted_df['ClinSig'].unique()

    unique_names = np.concatenate([unique_haplo, unique_effect, unique_rev, unique_clinsig]).tolist()

    print(unique_names)
    # Update the color of slices with label 'OTHER' to red
    for trace in fig.data:
        for i, label in enumerate(trace['ids']):
            actual_label = label.split('/')[-1]  # Always the last
            for j, uname in enumerate(unique_names):
                if uname in actual_label:
                    temp_color = list(trace['marker']['colors'])  # Convert tuple to list
                    temp_color[i] = color_palette[3 + j]
                    trace['marker']['colors'] = tuple(temp_color)  # Convert back to tuple

    # Save the chart as HTML file
    fig.write_html(f"{save_path}.html")

    # Show the chart
    fig.show()


def overall_success_pie(total_length, expectedlen, unexpectedlen, save_path):
    # Calculate the percentage of expected and unexpected cases
    exp_percentage = (expectedlen / total_length) * 100
    unexp_percentage = (unexpectedlen / total_length) * 100

    # Create labels and sizes for the pie chart
    labels = [f'Expected severity, {round(exp_percentage, 2)}', f'Unexpected severity, {round(unexp_percentage, 2)}']
    length_labels = [f'Expected severity, {expectedlen}', f'Unexpected severity, {unexpectedlen}']
    sizes = [round(exp_percentage, 2), round(unexp_percentage, 2)]

    # Plot the pie chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size for better visibility
    colors = [color_palette[1], color_palette[2]]
    explode = (0.1, 0)  # Explode the first slice
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140,
                                       textprops=dict(color="black"),
                                       colors=colors, shadow=True, explode=explode)

    # Add legends outside the pie chart
    plt.legend(wedges, length_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Set aspect ratio to be equal, and add a title
    plt.axis('equal')
    plt.title(f'Overall severity scores expectancy success rates', fontsize=16)

    # Save the pie chart to the specified path
    plt.savefig(save_path, bbox_inches='tight')

    # Display the pie chart
    plt.show()


# Visualize each subgroup statistics
def subgroup_expectancy_graph(subgroup_exp_rate, save_path):
    """
    param subgroup_exp_rate: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per tested subgroup
    param save_path: Output png path
    return: subgroup expectancy graph
    """
    # Print to log
    print(f'Start creating subgroups graph')
    subgroups = list(subgroup_exp_rate.keys())
    expected_counts, unexpected_counts = zip(*subgroup_exp_rate.values())

    # Filter out subgroups with expected + unexpected <= 0
    non_zero_subgroups = [subgroup for subgroup, (expected, unexpected) in subgroup_exp_rate.items() if
                          expected + unexpected > 0]

    # Calculate expectancy rates
    total_counts = [expected + unexpected for expected, unexpected in zip(expected_counts, unexpected_counts)]
    expectancy_rates = [(expected / total) * 100 if total > 0 else 0 for expected, total in
                        zip(expected_counts, total_counts)]

    # Filter expectancy rates for non-zero subgroups
    non_zero_expectancy_rates = [rate for subgroup, rate in zip(subgroups, expectancy_rates) if
                                 subgroup in non_zero_subgroups]

    # Partially decode the graphs encodings
    decoding_subgroups = list(graph_filter_interp(filter) for filter in non_zero_subgroups)

    # Set up the figure for the bar graph
    plt.figure(figsize=(12, 6))

    # Create the bar graph for each subgroup
    plt.bar(decoding_subgroups, non_zero_expectancy_rates, color=color_palette, edgecolor='black')

    # Add labels and title
    plt.xlabel('Subgroup')
    plt.ylabel('Expectancy Rate (%)')
    plt.title('Subgroup Expectancy Rates')

    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=90)

    # Display the expectancy rates on top of each bar
    for i, rate in enumerate(non_zero_expectancy_rates):
        plt.text(i, rate / 2, f'{rate:.2f}%', ha='center', va='bottom', color='black', fontweight='bold', rotation=-90)

    # Show the plot
    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot as an image
    plt.show()
    plt.close()  # Close the plot to release resources

    print(f'Finished creating subgroups graphs, results can be found at {save_path}')


# Visualize statistics based on effect category
def effect_expectancy_graph(effect_group_filters, save_path):
    """
    param effect_group_filters: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per effect type
    param save_path: Output png path
    return: effect expectancy graph
    """
    # Print to log
    print(f'Start creating effects graph')
    # Calculate success ratios and plot for effect groups
    plt.figure(figsize=(8, 6))

    effect_group_labels = []
    success_ratios = []

    for effect, stats in effect_group_filters.items():
        total_variants = stats[0] + stats[1]
        if total_variants > 0:
            success_ratio = (stats[0] / total_variants) * 100
            effect_group_labels.append(f'{effect}\n {total_variants} variants')
            success_ratios.append(success_ratio)

    plt.bar(effect_group_labels, success_ratios, color=color_palette, edgecolor='black')

    plt.xlabel('Effect Groups')
    plt.ylabel('Success Ratio (%)')
    plt.title('Success Ratio for Effect Groups')

    # Display the expectancy rates on top of each bar
    for i, rate in enumerate(success_ratios):
        plt.text(i, rate / 2, f'{rate:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

    print(f'Finished creating effects graph, results can be found at {save_path}')


def clnsig_effect_expectancy_graph(clnsig_effect_benign, clnsig_effect_pathogenic, save_path):
    """
    param clnsig_effect_benign^pathogenic: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per effect type
    param save_path: Output png path
    return: effect expectancy graph for benign and pathogenic seperatley
    """
    # Print to log
    print(f'Start creating effects graph')
    # Create save paths
    benign_save_path = save_path.replace('.png', '_benign.png')
    pathogenic_save_path = save_path.replace('.png', '_pathogenic.png')
    # Calculate success ratios and plot for effect groups
    plt.figure(figsize=(8, 6))

    effect_group_labels = []
    success_ratios = []

    for effect, stats in clnsig_effect_benign.items():
        total_variants = stats[0] + stats[1]
        print(f"Effect: {effect}, Stats: {stats}, Total Variants: {total_variants}")
        if total_variants > 0:
            success_ratio = (stats[0] / total_variants) * 100
            effect_group_labels.append(f'{effect}\n {total_variants} variants')
            success_ratios.append(success_ratio)

    plt.bar(effect_group_labels, success_ratios, color=color_palette, edgecolor='black')

    plt.xlabel('Effect Groups')
    plt.ylabel('Success Ratio (%)')
    plt.title('Success Ratio for Effect Groups for benign variants')

    # Display the expectancy rates on top of each bar
    for i, rate in enumerate(success_ratios):
        plt.text(i, rate / 2, f'{rate:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(benign_save_path)
    plt.show()
    plt.close()

    # Calculate success ratios and plot for effect groups
    plt.figure(figsize=(8, 6))

    effect_group_labels = []
    success_ratios = []

    for effect, stats in clnsig_effect_pathogenic.items():
        total_variants = stats[0] + stats[1]
        print(f"Effect: {effect}, Stats: {stats}, Total Variants: {total_variants}")
        if total_variants > 0:
            success_ratio = (stats[0] / total_variants) * 100
            effect_group_labels.append(f'{effect}\n {total_variants} variants')
            success_ratios.append(success_ratio)

    plt.bar(effect_group_labels, success_ratios, color=color_palette, edgecolor='black')

    plt.xlabel('Effect Groups')
    plt.ylabel('Success Ratio (%)')
    plt.title('Success Ratio for Effect Groups for pathogenic variants')

    # Display the expectancy rates on top of each bar
    for i, rate in enumerate(success_ratios):
        plt.text(i, rate / 2, f'{rate:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(pathogenic_save_path)
    plt.show()
    plt.close()

    print(
        f'Finished creating effects graphs splitted for benign and pathogenic,\n results can be found at {benign_save_path}, {pathogenic_save_path}')


# Visualize Haploinsufficiency statistics
def hi_expectancy_graph(hi_statistics, save_path):
    """
    param hi_statistics: Dictionary of tuples ( Expected, Unexpected) for calculating expectancy rates per haploinsufficiency group
    param save_path: Output png path
    return: Haploinsufficiency expectancy graph
    """
    # Print to log
    print(f'Start creating haploinsufficiency graph')
    # Calculate success ratios and plot for effect groups
    plt.figure(figsize=(8, 6))

    hi_group_labels = []
    success_ratios = []

    for hi, stats in hi_statistics.items():
        total_variants = stats[0] + stats[1]
        if total_variants > 0:
            success_ratio = (stats[0] / total_variants) * 100
            hi_group_labels.append(f'{hi}\n {total_variants} variants')
            success_ratios.append(success_ratio)

    plt.bar(hi_group_labels, success_ratios, color=color_palette, edgecolor='black')

    plt.xlabel('Haploinsufficiency Groups')
    plt.ylabel('Success Ratio (%)')
    plt.title('Success Ratio for Effect Groups')

    # Display the expectancy rates on top of each bar
    for i, rate in enumerate(success_ratios):
        plt.text(i, rate / 2, f'{rate:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

    print(f'Finished creating haploinsufficiency graph, results can be found at {save_path}')


# Visualize TPR FPR in auroc dot graph
def auroc_dot(auroc, save_path):
    """
    Plot a smooth AUC-ROC curve in the AUROC space.

    Parameters:
    - auroc (dict): Dictionary containing true positive (tp), true negative (tn),
                    false positive (fp), and false negative (fn) counts.
    - save_path (str): Save the plot to the specified path.

    Returns:
    None
    """
    # Calculate TPR and FPR
    tpr = auroc["tp"] / (auroc["tp"] + auroc["fn"])
    fpr = auroc["fp"] / (auroc["fp"] + auroc["tn"])

    # Plot the smooth curve
    plt.figure()

    # Plot the smooth curve segments
    plt.plot([0, fpr], [0, tpr], color=color_palette[-2], lw=2)  # Line from (0,0) to (fpr,tpr)
    plt.plot([fpr, 1], [tpr, 1], color=color_palette[-2], lw=2)  # Line from (fpr,tpr) to (1,1)

    # Mark the point (fpr, tpr) with a red dot
    plt.scatter(fpr, tpr, color=color_palette[2], marker='o', label='Threshold Point')

    # Annotate FPR and TPR values
    plt.annotate(f'FPR: {fpr:.2f}\nTPR: {tpr:.2f}', (fpr, tpr), textcoords="offset points", xytext=(0, 10), ha='center',
                 fontsize=8, color='black')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color=color_palette[-1], lw=2, linestyle='--')

    # Set the axis limits to [0, 1]
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add legend
    plt.legend()
    # Add legend with specified values
    legend_text = f'TP = High & Pathogenic, TN = Low & Benign\n' \
                  f' FP = High & Benign, FN = Low & Pathogenic\n' \
                  f'TP = {auroc["tp"]}\nTN = {auroc["tn"]}\nFP = {auroc["fp"]}\nFN = {auroc["fn"]}\n' \
                  f'Specificity =  TN / (TN + FP) = {auroc["tn"] / (auroc["tn"] + auroc["fp"]):.2f}\n' \
                  f'Sensitivity  = TPR = TP / (TP + FN) = {tpr:.2f}\n' \
                  f'FPR = FP / (FP + TN) = {fpr:.2f}\n'
    plt.text(0.60, 0.40, legend_text, fontsize=7, verticalalignment='top')

    # Add title
    plt.title('AUC-ROC')
    # Save the plot if a save path is provided
    plt.tight_layout()
    plt.savefig(save_path)

    # Display the plot
    plt.show()
    plt.close()

    return fpr, tpr


def severity_pie_chart(tsv_path, save_path):
    """
    param tsv_path: Input variants TSV curated file
    param save_path: Output png path
    return: Severity division pie chart
    """
    # Read TSV file into a Pandas DataFrame
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

    # Group data by the 'Severity' column and count occurrences
    severity_counts = df['Severity'].value_counts()

    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(severity_counts, labels=severity_counts.index, colors=color_palette,
                                       autopct=lambda p: f'{p:.1f}%\n({int(p * sum(severity_counts) / 100)})',
                                       startangle=140, wedgeprops=dict(width=0.4), textprops=dict(color="w"))

    # Draw a circle at the center to make it look like a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Adjust text properties for better visibility
    for text, autotext in zip(texts, autotexts):
        text.set_color('black')
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    # Set aspect ratio to be equal, and add a title
    plt.axis('equal')
    plt.title('Severity Distribution')

    # Save the pie chart to the specified path
    plt.savefig(save_path, bbox_inches='tight')

    # Display the pie chart
    plt.show()


def merge_categories(label, filter):
    label = str(label)
    filter = str(filter)
    if 'Pathogenic' in label and 'Likely' in label:
        return 'Likely Pathogenic'
    elif 'Pathogenic' in label:
        return 'Pathogenic'
    elif 'Benign' in label and 'Likely' in label:
        return 'Likely Benign'
    elif 'Benign' in label:
        return 'Benign'
    else:
        # Return likely benign or likely pathogenic based on filter for UncertainSignificance\Nan\NoSignificance as stated in ClinVar
        if any(num in filter for num in ['5', '6', '7', '8', '9', '10']):
            return 'Likely Benign'
        else:
            return 'Likely Pathogenic'


def clnsig_pie_chart(tsv_path, save_path):
    """
    param tsv_path: Input variants TSV curated file
    param save_path: Output png path
    return: ClinVar clinical significance division pie chart
    """
    # Read TSV file into a Pandas DataFrame
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

    # Merge categories
    # Apply the merge_categories function to 'ClinVar' column based on values from 'Filter' column
    df['ClinVar'] = df.apply(lambda row: merge_categories(row['ClinVar'], row['FILTER']), axis=1)

    # Group data by the 'Severity' column and count occurrences
    clnsig_counts = df['ClinVar'].value_counts()

    # Define preferred order of slices
    preferred_order = ['Benign', 'Likely Benign', 'Pathogenic', 'Likely Pathogenic']

    # Reorder the values based on preferred_order
    clnsig_counts = clnsig_counts.reindex(preferred_order)

    # Plot a pie chart if there are non-zero occurrences
    if not clnsig_counts.empty:
        plt.figure(figsize=(8, 8))
        wedges, _, autotexts = plt.pie(clnsig_counts, labels=clnsig_counts.index, colors=color_palette,
                                       autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100 * clnsig_counts.sum())})',
                                       startangle=140, wedgeprops=dict(width=0.4), textprops=dict(color="black"))

        # Adjust text properties for better visibility
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')

        # Set aspect ratio to be equal, and add a title
        plt.axis('equal')
        plt.title('ClinVar Clinical Significance Distribution')

        # Save the pie chart to the specified path
        plt.savefig(save_path, bbox_inches='tight')

        # Display the pie chart
        plt.show()
    else:
        print("No non-zero occurrences to plot. Skipping pie chart generation.")


# ALPHAMISSENSE visualization for unexpected missense variants
def alphamissense_visualize(var_tsv, save_path):
    type = var_tsv.split('_')[1]  # expected or unexpected
    # Read the TSV file into a DataFrame
    print("Reading unexpected TSV file...")
    print(f"Start processing {type} file for ALPHAMISSENSE distribution.")
    df = pd.read_csv(var_tsv, sep='\t')

    # Check if the required columns are present in the DataFrame
    required_columns = ['AM_Score']
    if not set(required_columns).issubset(df.columns):
        missing_columns = set(required_columns) - set(df.columns)
        raise KeyError(f"Missing columns in the CSV file: {missing_columns}")

    # Check data type of 'AM_Score' column
    if not pd.api.types.is_numeric_dtype(df['AM_Score']):
        raise TypeError("AM_Score column is not numeric.")

    # Truncate df to contain missense variants only
    df = df[df['FILTER'].str.contains('2', regex=False) | df['FILTER'].str.contains('7', regex=False)]
    num_rows = len(df)
    # Create histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
    plt.hist(df['AM_Score'], bins=bins, color='blue', edgecolor='black')
    plt.xlabel('ALPHAMISSENSE Values')
    plt.ylabel('Number of variants')
    plt.title(f'ALPHAMISSENSE Histogram For All {type} Missense Variants, Overall {num_rows} variants')
    plt.grid(False)
    plt.xticks(np.arange(0, 1.1, step=0.1), rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(save_path)
    plt.close()
    print(f"Finished creating AlphaMissense spread, results can be found at {save_path}")

    # For Benign only we take variants with High severity because this file is the unexpected variants file, same for pathogenic
    # Filter rows by severity levels Low and High
    low_severity = df[df['Severity'] == 'Low']
    high_severity = df[df['Severity'] == 'High']
    # extract matching dataframes
    benign_df = low_severity['AM_Score']
    benign_num_rows = len(benign_df)
    pathogenic_df = high_severity['AM_Score']
    pathogenic_num_rows = len(pathogenic_df)
    # Plot seperate graphs
    # Create histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
    plt.hist(benign_df, bins=bins, color='blue', edgecolor='black')
    plt.xlabel('ALPHAMISSENSE Values')
    plt.ylabel('Number of variants')
    plt.title(f'ALPHAMISSENSE Histogram For Benign {type} Missense Variants, Overall {benign_num_rows} variants')
    plt.grid(False)
    plt.xticks(np.arange(0, 1.1, step=0.1), rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_benign.png'))
    plt.close()
    print(
        f"Finished creating AlphaMissense benign spread, results can be found at {save_path.replace('.png', '_benign.png')}")
    # Create histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
    plt.hist(pathogenic_df, bins=bins, color='blue', edgecolor='black')
    plt.xlabel('ALPHAMISSENSE Values')
    plt.ylabel('Number of variants')
    plt.title(
        f'ALPHAMISSENSE Histogram For Pathogenic {type} Missense Variants, Overall {pathogenic_num_rows} variants')
    plt.grid(False)
    plt.xticks(np.arange(0, 1.1, step=0.1), rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_pathogenic.png'))
    plt.close()
    print(
        f"Finished creating AlphaMissense pathogenic spread, results can be found at {save_path.replace('.png', '_pathogenic.png')}")


# Present distribution of impact out of all Low unexpected variants
def impact_visualize(unexp_tsv, save_path):
    # Read the TSV file into a DataFrame
    print("Reading unexpected TSV file...")
    print("Start processing unexpected file for SNPEFF-IMPACT distribution.")
    df = pd.read_csv(unexp_tsv, sep='\t')

    # Check if the required columns are present in the DataFrame
    required_columns = ['IMPACT']
    if not set(required_columns).issubset(df.columns):
        missing_columns = set(required_columns) - set(df.columns)
        raise KeyError(f"Missing columns in the CSV file: {missing_columns}")

    # Truncate df to contain low variants only
    low_severity = df[df['Severity'] == 'Low']
    # Group data by the 'Severity' column and count occurrences
    impact_counts = low_severity['IMPACT'].value_counts()

    # Define preferred order of slices
    preferred_order = ['LOW', 'MODERATE', 'MODIFIER', 'HIGH']

    # Reorder the values based on preferred_order
    impact_counts = impact_counts.reindex(preferred_order)

    # Plot a pie chart if there are non-zero occurrences
    if not impact_counts.empty:
        plt.figure(figsize=(8, 8))
        wedges, _, autotexts = plt.pie(impact_counts, labels=impact_counts.index, colors=color_palette[1:],
                                       autopct=lambda pct: f'{pct:.1f}%({int(pct / 100 * impact_counts.sum())})',
                                       wedgeprops=dict(width=0.4), textprops=dict(color="black"))

        # Adjust text properties for better visibility
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')

        # Set aspect ratio to be equal, and add a title
        plt.axis('equal')
        plt.title('SNPEFF IMPACT Distribution Across Low Severity Unexpected Variants')

        # Save the pie chart to the specified path
        plt.savefig(save_path, bbox_inches='tight')

        # Display the pie chart
        plt.show()
    else:
        print("No non-zero occurrences to plot. Skipping pie chart generation.")
    print(f"Finished creating SNPEFF-IMPACT distribution pie chart, results can be found at {save_path}")


# TESTING FUNCTIONS
# determine effect qa test
def test_determine_effect_qa():
    """
    return: Excel file with QA effects mapping saved at "qa_matched_effects.xlsx"
    """
    lists = [['UTR_5_PRIME'], ['SYNONYMOUS_CODING'], ['NON_SYNONYMOUS_CODING'], ['CODON_DELETION'], ['FRAME_SHIFT'],
             ['INTRON'], ['INTRON', 'SPLICE_SITE_REGION'], ['SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['SYNONYMOUS_STOP'], ['UTR_3_PRIME'], ['START_GAINED', 'UTR_5_PRIME'], ['DOWNSTREAM'], ['EXON'],
             ['UPSTREAM'], ['EXON', 'SPLICE_SITE_REGION'], ['STOP_GAINED'], ['INTRON', 'SPLICE_SITE_DONOR'],
             ['INTRON', 'SPLICE_SITE_ACCEPTOR'], ['FRAME_SHIFT', 'NON_SYNONYMOUS_CODING'],
             ['SPLICE_SITE_REGION', 'STOP_GAINED'], ['INTERGENIC'], ['NON_SYNONYMOUS_CODING', 'SPLICE_SITE_REGION'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['SPLICE_SITE_REGION', 'UTR_3_PRIME'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'], ['FRAME_SHIFT', 'STOP_GAINED'],
             ['INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'], ['FRAME_SHIFT', 'SPLICE_SITE_REGION'],
             ['START_LOST'], ['NON_SYNONYMOUS_START'], ['CODON_INSERTION'],
             ['INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'], ['CODON_CHANGE_PLUS_CODON_DELETION'],
             ['CODON_INSERTION', 'STOP_GAINED'], ['FRAME_SHIFT', 'STOP_LOST', 'UTR_3_PRIME'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION'], ['CODON_CHANGE_PLUS_CODON_DELETION', 'SYNONYMOUS_CODING'],
             ['CODON_INSERTION', 'NON_SYNONYMOUS_CODING', 'STOP_GAINED'], ['FRAME_SHIFT', 'SYNONYMOUS_CODING'], [''],
             ['STOP_LOST'], ['SPLICE_SITE_REGION', 'UTR_5_PRIME'],
             ['EXON', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'TRANSCRIPT', 'UTR_5_PRIME'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['EXON', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['FRAME_SHIFT', 'START_LOST'],
             ['INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['FRAME_SHIFT_AFTER_CDS_END', 'UTR_3_PRIME'], ['FRAME_SHIFT', 'NON_SYNONYMOUS_CODING', 'STOP_GAINED'],
             ['CODON_INSERTION', 'SPLICE_SITE_REGION'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['FRAME_SHIFT', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'START_LOST', 'UTR_5_PRIME'],
             ['EXON_DELETED', 'FRAME_SHIFT', 'START_LOST'],
             ['INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'TRANSCRIPT', 'UTR_5_PRIME'],
             ['FRAME_SHIFT', 'INTRON', 'NON_SYNONYMOUS_START', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'START_LOST',
              'UTR_5_PRIME'], ['FRAME_SHIFT_BEFORE_CDS_START', 'UTR_5_PRIME'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'START_LOST'],
             ['SPLICE_SITE_REGION', 'START_GAINED', 'UTR_5_PRIME'],
             ['FRAME_SHIFT', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['CODON_INSERTION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION',
              'STOP_GAINED'], ['CODON_DELETION', 'SPLICE_SITE_REGION'],
             ['FRAME_SHIFT', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_REGION'],
             ['INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'START_LOST', 'UTR_5_PRIME'],
             ['FRAME_SHIFT', 'SPLICE_SITE_REGION', 'STOP_GAINED'], ['CODON_CHANGE_PLUS_CODON_DELETION', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'START_LOST', 'UTR_5_DELETED'],
             ['FRAME_SHIFT', 'START_LOST', 'UTR_5_PRIME'], ['CODON_DELETION', 'STOP_GAINED'],
             ['FRAME_SHIFT', 'STOP_LOST'], ['CODON_CHANGE_PLUS_CODON_DELETION', 'STOP_LOST'],
             ['FRAME_SHIFT', 'SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_DONOR'], ['CODON_DELETION', 'START_LOST', 'UTR_5_PRIME'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'NON_SYNONYMOUS_CODING', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'NON_SYNONYMOUS_CODING', 'STOP_GAINED'],
             ['FRAME_SHIFT', 'STOP_GAINED', 'SYNONYMOUS_CODING'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'TRANSCRIPT', 'UTR_3_PRIME'],
             ['EXON_DELETED', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['CODON_INSERTION', 'NON_SYNONYMOUS_CODING'], ['EXON_DELETED', 'SPLICE_SITE_REGION'],
             ['SPLICE_SITE_REGION', 'UTR_5_DELETED'], ['CODON_CHANGE_PLUS_CODON_DELETION', 'NON_SYNONYMOUS_CODING'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR'], ['CODON_DELETION', 'NON_SYNONYMOUS_CODING'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'SYNONYMOUS_CODING'], ['TRANSCRIPT', 'UPSTREAM', 'UTR_5_PRIME'],
             ['EXON', 'UPSTREAM'], ['CODON_CHANGE_PLUS_CODON_INSERTION', 'NON_SYNONYMOUS_CODING'],
             ['INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'TRANSCRIPT', 'UTR_3_PRIME'],
             ['CODON_DELETION', 'STOP_LOST', 'UTR_3_PRIME'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_DONOR',
              'SPLICE_SITE_REGION'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['FRAME_SHIFT', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_DELETION', 'NON_SYNONYMOUS_CODING', 'STOP_LOST', 'UTR_3_PRIME'],
             ['CODON_INSERTION', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION',
              'STOP_GAINED'], ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION',
                               'SYNONYMOUS_CODING'],
             ['CODON_DELETION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'START_LOST', 'UTR_5_DELETED'],
             ['INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['SPLICE_SITE_REGION', 'SYNONYMOUS_STOP'],
             ['EXON_DELETED', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION',
              'UTR_5_DELETED'], ['INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'SYNONYMOUS_CODING'],
             ['SPLICE_SITE_REGION', 'START_LOST'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'STOP_GAINED', 'SYNONYMOUS_CODING'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'SPLICE_SITE_REGION'], ['EXON_DELETED', 'FRAME_SHIFT'],
             ['CODON_DELETION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR',
              'SPLICE_SITE_REGION'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION',
              'SYNONYMOUS_CODING'], ['NON_SYNONYMOUS_START', 'SPLICE_SITE_REGION'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR',
              'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_DELETION', 'SYNONYMOUS_CODING'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED', 'SYNONYMOUS_CODING'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR',
              'SPLICE_SITE_REGION'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['FRAME_SHIFT', 'SPLICE_SITE_REGION', 'START_LOST', 'UTR_5_PRIME'],
             ['EXON_DELETED', 'SPLICE_SITE_REGION', 'UTR_5_DELETED'],
             ['CODON_DELETION', 'START_LOST', 'SYNONYMOUS_CODING', 'UTR_5_PRIME'],
             ['FRAME_SHIFT', 'SPLICE_SITE_REGION', 'START_LOST', 'SYNONYMOUS_CODING', 'UTR_5_PRIME'],
             ['EXON', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['EXON_DELETED', 'FRAME_SHIFT', 'START_LOST', 'UPSTREAM'],
             ['FRAME_SHIFT', 'NON_SYNONYMOUS_CODING', 'START_LOST'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_DONOR',
              'SPLICE_SITE_REGION'], ['INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_INSERTION', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_REGION'],
             ['EXON_DELETED', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'STOP_GAINED',
              'SYNONYMOUS_CODING'], ['FRAME_SHIFT', 'SPLICE_SITE_REGION', 'START_LOST'],
             ['FRAME_SHIFT_BEFORE_CDS_START', 'SPLICE_SITE_REGION', 'UTR_5_PRIME'],
             ['INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'TRANSCRIPT', 'UTR_5_DELETED'],
             ['CODON_CHANGE_PLUS_CODON_DELETION', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_REGION'],
             ['EXON_DELETED', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION'],
             ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'SPLICE_SITE_REGION'], ['GENE_FUSION'],
             ['CODON_DELETION', 'START_LOST', 'STOP_GAINED', 'UTR_5_PRIME'],
             ['EXON_DELETED', 'INTRON', 'SPLICE_SITE_DONOR', 'SPLICE_SITE_REGION', 'UPSTREAM', 'UTR_5_DELETED'],
             ['GENE_FUSION_REVERESE'], ['DOWNSTREAM', 'SPLICE_SITE_REGION', 'TRANSCRIPT', 'UTR_3_PRIME'],
             ['CODON_CHANGE_PLUS_CODON_INSERTION', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_REGION', 'STOP_GAINED'],
             ['FRAME_SHIFT', 'INTRON', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION', 'STOP_LOST', 'SYNONYMOUS_CODING',
              'UTR_3_PRIME'], ['CODON_DELETION', 'INTRON', 'SPLICE_SITE_DONOR'],
             ['CODON_INSERTION', 'INTRON', 'NON_SYNONYMOUS_CODING', 'SPLICE_SITE_ACCEPTOR', 'SPLICE_SITE_REGION'],
             ['EXON_DELETED', 'SPLICE_SITE_REGION', 'UPSTREAM', 'UTR_5_DELETED'],
             ['EXON', 'SPLICE_SITE_REGION', 'UPSTREAM'],
             ['SPLICE_SITE_REGION', 'TRANSCRIPT', 'UPSTREAM', 'UTR_5_DELETED']]

    result_lists = [(lst, determine_effect_qa(lst)) for lst in lists]
    # Save the result to an Excel file
    # Create a DataFrame from the result_lists
    df = pd.DataFrame(result_lists, columns=['EFFECT', 'SUBGROUP'])

    # Save the DataFrame to an Excel file
    df.to_excel('qa_matching_effects.xlsx', index=False)
    flattened_list = [word for sublist in lists for word in sublist]

    # Get unique elements
    unique_words = list(set(flattened_list))
    print(f'All unique EFFECTS:{unique_words}')


def split_file_by_filter(input_file):
    """
    param input_file: Input TSV curated file
    return: TSV file per FILTER type
    """
    # Define output file paths
    output_paths = []

    # Print start message
    print(f'Started splitting file: {input_file}')

    # Read the entire file
    df = pd.read_csv(input_file, sep='\t')

    # Get unique values in the 'FILTER' column
    unique_filters = df['FILTER'].unique()

    for value in unique_filters:
        # Print start message for each filter
        print(f'Starting splitting for filter: {value}')

        # Create a filtered DataFrame based on the 'FILTER' value
        filtered_df = df[df['FILTER'] == value]

        # Define the output file name based on the 'FILTER' value
        output_file = f"{input_file.replace('.txt', f'_{value}.txt')}"
        output_paths.append(output_file)

        # Write the filtered DataFrame to a new TSV file
        filtered_df.to_csv(output_file, sep='\t', index=False)

        # Print completion message for each filter
        print(f'Finished splitting for filter: {value}')

    # Print completion message for all filters
    print(f'Finished splitting for all filters')

    return output_paths


def graph_filter_interp(filter):
    """
    param filter: FILTER, i.e A1SS
    return:  FILTER decoded
    """
    # Define interpretation of the subgroup
    subg = ''

    # Parse haploinsufficiency
    if 'A' in filter:
        subg += 'HI0 '
    elif 'B' in filter:
        subg += 'HI3 '
    elif 'C' in filter:
        subg += 'HI40 '
    # Parse effect group and clinical significance
    if '10' in filter:
        subg += 'OTHER B  '
    elif '1' in filter:
        subg += 'LOF P  '
    elif '2' in filter:
        subg += 'MISSENSE P  '
    elif '3' in filter:
        subg += 'SPLICE P  '
    elif '4' in filter:
        subg += 'START P '
    elif '5' in filter:
        subg += 'OTHER P '
    elif '6' in filter:
        subg += 'LOF B '
    elif '7' in filter:
        subg += 'MISSENSE B '
    elif '8' in filter:
        subg += 'SPLICE B '
    else:
        if '9' in filter:
            subg += 'START B '
    # Parse review status
    if 'SS' in filter:
        subg += '1s'
    elif 'DS' in filter:
        subg += '2s+'
    return subg


### THRESHOLD ASSESSMENT
def assessment():
    """
    This subprogram is initiated after "Assess Thresholds" button was activated
    It will test the severity thresholds distribution conditions, cutoffs and requirements.
    return: Retagged severity curated TSV file
    """
    # Define statistic threshold cutoffs
    # High stand-alone cutoffs
    # high_stand_alone_Ada_cutoff = 0.6
    # high_stand_alone_Rf_cutoff = 0.6
    high_stand_alone_SAIS_cutoff = 0.9
    high_stand_alone_CADDP_cutoff = 25
    # Medium stand-alone cutoffs
    med_stand_alone_Ada_cutoff = 0.6
    med_stand_alone_Rf_cutoff = 0.6
    med_stand_alone_SAIS_cutoff = 0.9
    med_stand_alone_CADDP_cutoff = 25  # 25

    # Cutoffs for CADDP
    high_CADDP_cutoff = 15
    med_CADDP_cutoff = 8

    # Max cutoff we test for smaller frequencies than this SIFT cutoff
    high_SIFT_cutoff = 0.05

    # Splice ai SAIS score
    high_SAIS_cutoff = 0.6
    med_SAIS_cutoff = 0.4

    # for both GERP_NR and GERP_RS
    high_GERP_cutoff = 2

    high_PHYLOP_cutoff = 0

    ## ADDED AM CUTOFF FOR TESTING
    high_AM_cutoff = 0.89999

    # Define conditions for stand-alone high severity, at least 2 of the following
    standalone_high_conditions = {
        # Removed for test
        # 'AdaScore': lambda x: float(x) > high_stand_alone_Ada_cutoff,
        # 'RfScore': lambda x: float(x) > high_stand_alone_Rf_cutoff,
        'SAIS': lambda x: float(x) >= high_stand_alone_SAIS_cutoff,
        'CADDP': lambda x: float(x) >= high_stand_alone_CADDP_cutoff
    }

    standalone_medium_conditions = {  # one of the conditions
        'AdaScore': lambda x: float(x) > med_stand_alone_Ada_cutoff,
        'RfScore': lambda x: float(x) > med_stand_alone_Rf_cutoff,
        'SAIS': lambda x: float(x) >= med_stand_alone_SAIS_cutoff,
        'CADDP': lambda x: float(x) >= med_stand_alone_CADDP_cutoff,
        # ADD TO TEST
        # 'LRT_PRED': lambda x: x == 'D',
        # 'MUTATIONTASTER_PRED': lambda x: x == 'D' or x == 'A',
        # 'SIFT_SCORE': lambda x: float(x) < high_SIFT_cutoff,
        # 'GERP_NR': lambda x: float(x) > high_GERP_cutoff,
        # 'GERP_RS': lambda x: float(x) > high_GERP_cutoff,
        # 'PHYLOP': lambda x: float(x) > high_PHYLOP_cutoff,
    }

    lof_conditions = {
        'High': {
            'CADDP': lambda x: pd.isna(x) or float(x) >= high_CADDP_cutoff,
        },
        'Medium': {
            'CADDP': lambda x: pd.isna(x) or float(x) >= med_CADDP_cutoff,
        },
    }

    missense_conditions = {
        'High': {  # all of the predictions
            'CADDP': lambda x: pd.isna(x) or float(x) >= high_CADDP_cutoff,
            'SIFT_SCORE': lambda x: float(x) <= high_SIFT_cutoff,
            'LRT_PRED': lambda x: x == 'D',
            'MUTATIONTASTER_PRED': lambda x: x == 'D' or x == 'A',
            'AM_Score': lambda x: pd.isna(x) or x >= high_AM_cutoff,
        },
        'Medium': {  # one of the predictions
            'CADDP': lambda x: pd.isna(x) or float(x) >= high_CADDP_cutoff,
            'SIFT_SCORE': lambda x: x <= high_SIFT_cutoff,
            'LRT_PRED': lambda x: x == 'D',
            'MUTATIONTASTER_PRED': lambda x: x == 'D' or x == 'A',
            'SAIS': lambda x: float(x) > high_SAIS_cutoff,
            'GERP_NR': lambda x: float(x) > high_GERP_cutoff,
            'GERP_RS': lambda x: float(x) > high_GERP_cutoff,
            'PHYLOP': lambda x: float(x) > high_PHYLOP_cutoff,
            'AM_Score': lambda x: x >= high_AM_cutoff,
        },
    }

    splice_conditions = {
        'High': {
            'SAIS': lambda x: pd.isna(x) or float(x) > high_SAIS_cutoff,
        },
        'Medium': {
            'SAIS': lambda x: pd.isna(x) or float(x) > med_SAIS_cutoff,
        },
    }
    # START AND OTHER WILL BE TESTED UNDER STAND-ALONE CONDITIONS ONLY

    # Define effect-based conditions
    effect_based_conditions = {
        'LOF': lof_conditions,
        'Missense': missense_conditions,
        'Splice': splice_conditions,
    }

    # Define relevant fields for each category
    standalone_high_fields = ['SAIS', 'CADDP']  # 'AdaScore', 'RfScore',
    # Test
    # standalone_medium_fields = ['SAIS', 'CADDP', 'LRT_PRED', 'MUTATIONTASTER_PRED', 'GERP_NR', 'GERP_RS', 'PHYLOP', 'SIFT_SCORE',]
    standalone_medium_fields = ['SAIS', 'CADDP','AdaScore', 'RfScore']
    lof_fields = ['CADDP']
    missense_fields = ['CADDP', 'SIFT_SCORE', 'LRT_PRED', 'MUTATIONTASTER_PRED', 'SAIS', 'GERP_NR', 'GERP_RS', 'PHYLOP','AM_Score']
    splice_fields = ['SAIS']


    # Tag severity again for all variants and make sure the unexpected and expected are identical
    def thresholds_reseverity(tsv_path, tsv_refiltered_path):
        """
        param tsv_path: Curated TSV variant file
        param tsv_refiltered_path: Output path for retagged severity
        return: TSV file with severity values tagged according to thresholds instead of the QA
        """
        # Print to log
        print(f'Start tagging severity based on thresholds conditions in the file {tsv_path} \n')
        # Read TSV file into a Pandas DataFrame
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

        # Filter rows where ManeStatus is 0 or 1, extra test should keep the file untouched
        df_filtered = df[df['ManeStatus'].isin([0, 1])]

        # Apply the reseverity function to each row
        df_filtered['Severity'] = df_filtered.apply(reseverity, axis=1)

        # Write the filtered DataFrame to a new TSV file
        df_filtered.to_csv(tsv_refiltered_path, sep='\t', index=False)

        print(f"Processed data saved to: {tsv_refiltered_path}")

    # subfunction to return medium for specific effects:
    def med_effects(variant):
        """
        param variant: Variant row from the TSV file
        return: Medium if the variant effect matched the med_effect_list else it returns not
        """
        med_effect_list = ["CODON_DELETION", "CODON_INSERTION", "START_GAINED", "START_LOST", "STOP_LOST",
                           "NON_SYNONYMOUS_START"]
        for eff in med_effect_list:
            if eff in variant['EFFECT']:
                return 'Medium'
        return 'not'

    # Subfunction for reseverity
    def group_effects(variant):
        """
        param variant: Variant row from the TSV file
        return: Matched effect to the variant according to the EFFECT column and not the filter or Molecular consequence
        """
        # Determines matching Effect group, LOF > MISSENSE > SPLICE > STRTSTP > OTHER
        effect_mapping = {
            "OTHER": ["FRAME_SHIFT_AFTER", "FRAME_SHIFT_BEFORE"],
            "LOF": ["STOP_GAINED", "SPLICE_SITE_DONOR", "SPLICE_SITE_ACCEPTOR"],
            "Missense": ["NON_SYNONYMOUS_CODING"],
            # "SPLICE": ["SPLICE_SITE_REGION", "SYNONYMOUS_CODING"],
            "Start": ["START_GAINED", "START_LOST", "STOP_LOST"]
        }

        # Create a list to store matched categories
        matched_categories = []

        # Define current variants effects
        effects = [word.strip() for word in variant['EFFECT'].split(',')] if ',' in variant['EFFECT'] else [
            str(variant['EFFECT'])]

        # Check each effect and append the matching category to the list
        for effect in effects:
            if pd.notna(effect):  # Skip nan values within the list
                for category, keywords in effect_mapping.items():
                    # Make sure FRAME_SHIFT taken as a full expression and not as a substring
                    if category == "LOF" and effect == "FRAME_SHIFT":
                        matched_categories.append(category)
                    elif any(keyword.lower() in effect.lower() for keyword in keywords):
                        matched_categories.append(category)

        # Check for splice site region + non_synonymous_coding or synonymous_coding
        syn = 0
        spl = 0
        for effect in effects:
            # splice site region
            if "SPLICE_SITE_REGION" in effect:
                spl = 1
            # non_synonymous_coding or synonymous_coding
            if "SYNONYMOUS_CODING" in effect:
                syn = 1
        if syn == 1 and spl == 1:
            matched_categories.append("Splice")

        # Return Other if no subgroup was found based on the unique keywords
        if len(matched_categories) == 0:
            return ["OTHER"]
        # Else, return all matching subgroups effects as a list
        return matched_categories

    # Subfunction for thresholds_reseverity that maps for a row its effect according to the statistics thresholds
    # will check for grouped effect conditions
    def reseverity(variant):
        """
        param variant: Variant row from the TSV file
        return: New severity value for the variant based on the thresholds defined in this program
        """
        # Parse all possible effects subgroups
        variant_effects = group_effects(variant)
        # Define severity priority
        severities_found = []
        # Check for most severe option across all effects, if High found or Med effects found return appropriately
        for variant_effect in variant_effects:
            # Check for stand alone High, needs at least 2 conditions from the stand-alone high conditions
            if sum(condition(variant[field]) for field, condition in standalone_high_conditions.items()) >= 2:
                return 'High'

            # Check for med - effects
            if variant_effect == 'Start' or variant_effect == 'OTHER' or variant_effect == 'Splice':
                ans = med_effects(variant)
                if ans == 'Medium':
                    severities_found.append("Medium")

            # Check for high according to effect needs, needs all the conditions
            if variant_effect != 'OTHER' and variant_effect != 'Start':
                for effect, conditions in effect_based_conditions.items():
                    if effect == variant_effect:
                        if all(condition(variant[field]) for field, condition in conditions['High'].items()):
                            return 'High'

            # Check for medium stand-alone high, needs at least one of the conditions
            if any(condition(variant[field]) for field, condition in standalone_medium_conditions.items()):
                severities_found.append('Medium')

            # Check for medium according to effect, needs at least one of the conditions
            elif variant_effect != 'OTHER' and variant_effect != 'Start':
                for effect, conditions in effect_based_conditions.items():
                    if variant_effect == effect:
                        if variant_effect == 'Missense':
                            # Check if both 'LRT_PRED' and 'MUTATIONTASTER_PRED' are both NaN
                            if all(pd.isna(variant[field]) for field in ['LRT_PRED', 'MUTATIONTASTER_PRED']):
                                severities_found.append('Medium')
                        if any(condition(variant[field]) for field, condition in
                               conditions['Medium'].items()):
                            severities_found.append('Medium')
        # If the function has not returned so far, severities found is either empty or contains Medium
        if len(severities_found) > 0:
            return 'Medium'
        # Else map as low
        return 'Low'

    def thresholds_show_difference_and_export(cv_tsv, reseverity_tsv, output_excel):
        """
        param cv_tsv: Curated TSV file
        param reseverity_tsv: Retagged severity curated TSV file
        param output_excel:  output path
        return: A file containing all rows tagged with different severities
                and the conditions lead to the retagged severity
        """
        # Read the TSV files into Pandas DataFrames
        df_cv = pd.read_csv(cv_tsv, sep='\t', encoding='utf-8')
        df_reseverity = pd.read_csv(reseverity_tsv, sep='\t', encoding='utf-8')

        # Assuming both DataFrames have a 'Severity' column
        severity_column = 'Severity'
        if severity_column not in df_cv.columns or severity_column not in df_reseverity.columns:
            print(f"Error: Both DataFrames must have a '{severity_column}' column.")
            return

        # Identify rows with different severities
        different_severities_mask = df_cv[severity_column] != df_reseverity[severity_column]

        # Extract rows with different severities
        different_severities_df = df_cv[different_severities_mask]

        # If there are no differences, export an empty Excel file with headers only
        if len(different_severities_df) == 0:
            print(f'No differences in severities')
            return 0

        # Add a new column for Retagged Severity in the extracted DataFrame
        different_severities_df['Retagged Severity'] = df_reseverity[severity_column]

        # Export the rows with different severities to an Excel file
        different_severities_df.to_excel(output_excel, index=False, engine='openpyxl', header=True)

        print(f'Number of rows with different severities: {len(different_severities_df)}')
        print(f'Different severities exported to: {output_excel}')

        # Return the count
        return len(different_severities_df)

    def diff_reseverity_reason(variant):
        """
        param variant: Variant row from the TSV file
        return: Same like reseverity, only with which conditions made it passed
        """
        # Define severity priority
        severities_found = []
        variant_effects = group_effects(variant)
        for variant_effect in variant_effects:
            # Check for stand alone High, needs at least 2 conditions from the stand-alone high conditions
            if sum(condition(variant[field]) for field, condition in standalone_high_conditions.items()) >= 2:
                return f'High Stand-alone, {variant_effect}'

            # Check for med - effects
            if variant_effect == 'Start' or variant_effect == 'OTHER' or variant_effect == 'Splice':
                ans = med_effects(variant)
                if ans == 'Medium':
                    severities_found.append('Medium, med - effects')

            # Check for high according to effect needs, needs all the conditions
            if variant_effect != 'OTHER' and variant_effect != 'Start':
                for effect, conditions in effect_based_conditions.items():
                    if effect == variant_effect:
                        if all(condition(variant[field]) for field, condition in conditions['High'].items()):
                            return f'High Effect-based {variant_effect}'

            # Check for medium stand-alone high, needs at least one of the conditions
            if any(condition(variant[field]) for field, condition in standalone_medium_conditions.items()):
                severities_found.append(f'Medium Stand-alone {variant_effect}')

            # Check for medium according to effect, needs at least one of the conditions
            elif variant_effect != 'OTHER' and variant_effect != 'Start':
                for effect, conditions in effect_based_conditions.items():
                    if variant_effect == effect:
                        if variant_effect == 'Missense':
                            # Check if both 'LRT_PRED' and 'MUTATIONTASTER_PRED' are both NaN
                            if all(pd.isna(variant[field]) for field in ['LRT_PRED', 'MUTATIONTASTER_PRED']):
                                severities_found.append(f'Medium Effect-based {variant_effect}')
                        if any(condition(variant[field]) for field, condition in
                               conditions['Medium'].items()):
                            severities_found.append(f'Medium Effect-based {variant_effect}')
        if len(severities_found) > 0:
            return severities_found[0]
        # Else map as low
        return f'Low, Stand-alone {variant_effects}'

    def add_severity_condition_to_diff(diff_excel_path):
        """
        param diff_excel_path: Excel file containing unmatched severity tag rows
        return: Updated Excel file with "Severity Condition" column containing the reason behind the severity tagging
        """
        # Read the diff.xlsx file into a Pandas DataFrame
        df_diff = pd.read_excel(diff_excel_path, engine='openpyxl')

        # Assuming 'Severity' and 'Retagged Severity' columns are already present in the DataFrame
        severity_column = 'Severity'
        retagged_severity_column = 'Retagged Severity'

        if severity_column not in df_diff.columns or retagged_severity_column not in df_diff.columns:
            print(
                f"Error: Both '{severity_column}' and '{retagged_severity_column}' columns must be present in the DataFrame.")
            return

        # Add a new column for Severity Condition using the diff_reseverity_reason function
        df_diff['Severity Condition'] = df_diff.apply(diff_reseverity_reason, axis=1)

        # Save the updated DataFrame to a new Excel file
        df_diff.to_excel(diff_excel_path, index=False, engine='openpyxl', header=True)

        print(f'Severity Condition added to the diff file and saved to: {diff_excel_path}')

    # Current main untill moves this code to optimize button in the tkinter of severity_validation_visualization:

    # Define paths and variables
    cv_tsv = 'qa_clinvar_refiltered.txt'
    reseverity_tsv = 'reseverity.txt'
    difference_excel = 'diff.xlsx'

    # Functions
    thresholds_reseverity(cv_tsv, reseverity_tsv)
    x = thresholds_show_difference_and_export(cv_tsv, reseverity_tsv, difference_excel)
    if x != 0:
        add_severity_condition_to_diff(difference_excel)


### ANALYZE UNEXPECTED VARIANTS
def analyze_unexpected():
    # Define colors palette for all graphs
    color_palette = ['#c4bbbb', '#d6edb7', '#cf665b', '#f2f0f7', '#bcbddc',
                     '#fcfbfd', '#dadaeb', '#9e9ac8', '#ebc5dd',
                     '#c4b0d9', '#e5d8bd', '#d2e3e7', '#f5e6e8', '#e3d9e1',
                     '#f1eadf', '#f8d39b', '#eeb263', '#df941b',
                     '#c36000', '#b04800', '#84b7cf', '#5998b5', '#7a7dff', '#3773b3', '#5744fc', '#9fa14a']

    def find_variants_cv_info(cv_tsv, db):
        """
        This function will search each variant according to CHROM POS ID REF ALT from the unexpected variants file, i.e., cv_tsv in the database
        and will add a column CLNREVSTAT in the cv_tsv and fill it up according to the variants CLNREVSTAT value in the database
        param cv_tsv: Input file path of unexpected variants
        param db: Database (VCF file) to pull variants data from
        return: None
        """
        # Read the unexpected variants file into a DataFrame
        cv_df = pd.read_csv(cv_tsv, sep='\t')

        # Initialize a dictionary to store CLNREVSTAT values
        clnrevstat_dict = {}

        # Read the entire VCF file into memory
        with open(db, 'r') as vcf_file:
            vcf_lines = vcf_file.readlines()

        # Populate CLNREVSTAT dictionary based on CHROM, POS, REF, and ALT
        for line in vcf_lines:
            if not line.startswith('#'):
                vcf_fields = line.strip().split('\t')
                chrom = vcf_fields[0]
                pos = vcf_fields[1]
                ref = vcf_fields[3]
                alt = vcf_fields[4]
                info_field = vcf_fields[7]
                if 'CLNREVSTAT=' in info_field:
                    clnrevstat_value = [x.split('=')[1] for x in info_field.split(';') if x.startswith('CLNREVSTAT=')][
                        0]
                    clnrevstat_dict[(chrom, pos, ref, alt)] = clnrevstat_value

            # Add CLNREVSTAT values to the cv_tsv DataFrame
        for index, row in cv_df.iterrows():
            variant_key = (str(row['Chromosome']), str(row['PositionStart']), str(row['REF']), str(row['ALT']))
            cv_df.at[index, 'CLNREVSTAT'] = clnrevstat_dict.get(variant_key, 'NA')

        # Make a list of variants with 'NA' in the CLNREVSTAT column
        na_variants = cv_df[cv_df['CLNREVSTAT'] == 'NA']

        # Search for variants with 'NA' in the CLNREVSTAT column based on CHROM and POS only
        for idx, row in na_variants.iterrows():
            chrom = row['Chromosome']
            pos = row['PositionStart']
            print(f'NA variant check: {chrom}:{pos}')
            for line in vcf_lines:
                if not line.startswith('#'):
                    vcf_fields = line.strip().split('\t')
                    if str(vcf_fields[0]) == str(chrom) and str(vcf_fields[1]) == str(pos):
                        info_field = vcf_fields[7]
                        if 'CLNREVSTAT=' in info_field:
                            clnrevstat_value = \
                                [x.split('=')[1] for x in info_field.split(';') if x.startswith('CLNREVSTAT=')][0]
                            cv_df.at[idx, 'CLNREVSTAT'] = clnrevstat_value
                            print(
                                f'NA variant CLNREVSTAT found based on chrom and pos only, and updated to {clnrevstat_value}')
                            break  # Stop searching once a matching variant is found

        # Write the updated cv_tsv DataFrame back to the file
        output_file_path = os.path.splitext(cv_tsv)[0] + '_with_CLNREVSTAT.txt'
        cv_df.to_csv(output_file_path, sep='\t', index=False)

        print(f'Finished parsing variants from the database {db}.\n'
              f'Relevant file can be found at {output_file_path}.\n')

    def clvrevstat_distribution_graph(variants_tsv, save_path):
        """
        Generate bar graphs representing the distribution of CLVREVSTAT across different CLNREVSTAT types
        for severity levels Low and High.

        :param variants_tsv: Input TSV file containing variant data
        :param save_path: Output png path
        :return: None
        """
        # Read the TSV file into a DataFrame
        df = pd.read_csv(variants_tsv, sep='\t')

        # Check if the required columns are present in the DataFrame
        required_columns = ['Severity', 'CLNREVSTAT']
        if not set(required_columns).issubset(df.columns):
            missing_columns = set(required_columns) - set(df.columns)
            raise KeyError(f"Missing columns in the TSV file: {missing_columns}")

        # Filter rows by severity levels Low and High
        low_severity = df[df['Severity'] == 'Low']
        high_severity = df[df['Severity'] == 'High']

        # Count CLVREVSTAT occurrences for each CLNREVSTAT type for severity level Low
        low_counts = low_severity['CLNREVSTAT'].value_counts()

        # Count CLVREVSTAT occurrences for each CLNREVSTAT type for severity level High
        high_counts = high_severity['CLNREVSTAT'].value_counts()

        # Plot bar graphs for severity level Low
        plt.figure(figsize=(10, 6))
        low_counts.plot(kind='bar', color=color_palette)
        plt.xlabel('CLNREVSTAT Types')
        plt.ylabel('Number of Variants')
        plt.title('Clinvar review status distribution for Pathogenic variants tagged with Severity Level: Low')
        plt.xticks(rotation=45)

        # Add group sizes (actual numbers) on top of the bars for severity level Low
        for i, v in enumerate(low_counts):
            plt.text(i, v + 0.1, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_low.png'))
        plt.close()

        # Plot bar graphs for severity level High
        plt.figure(figsize=(10, 6))
        high_counts.plot(kind='bar', color=color_palette)
        plt.xlabel('CLNREVSTAT Types')
        plt.ylabel('Number of Variants')
        plt.title('Clinvar review status distribution for Benign variants tagged with Severity Level: High')
        plt.xticks(rotation=45)

        # Add group sizes (actual numbers) on top of the bars for severity level High
        for i, v in enumerate(high_counts):
            plt.text(i, v + 0.1, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_high.png'))
        plt.close()

        print(f'Finished creating CLNREVSTAT distribution graphs, results can be found at {save_path}')

    def af_histogram(cv_path, save_path):
        """
        Generate histograms of the AC_AF field values from the input file for high and low values separately
        and save them to the specified path.

        :param cv_path: Path to the input CSV file containing variant data
        :param save_path: Path to save the histogram plots
        :return: None
        """
        # Read the CSV file into a DataFrame
        print("Reading CSV file...")
        df = pd.read_csv(cv_path, sep='\t')

        # Check if the required columns are present in the DataFrame
        required_columns = ['AC_AF', 'Severity']
        if not set(required_columns).issubset(df.columns):
            missing_columns = set(required_columns) - set(df.columns)
            raise KeyError(f"Missing columns in the CSV file: {missing_columns}")

        print("Filtering data by severity levels...")
        # Filter rows by severity levels Low and High
        low_severity = df[df['Severity'] == 'Low']
        high_severity = df[df['Severity'] == 'High']

        # Round AC_AF values to 4 decimal places
        low_severity['AC_AF'] = low_severity['AC_AF'].round(4)
        high_severity['AC_AF'] = high_severity['AC_AF'].round(4)

        # Get unique AC_AF values and their counts for low and high severity
        low_values, low_counts = np.unique(low_severity['AC_AF'], return_counts=True)
        high_values, high_counts = np.unique(high_severity['AC_AF'], return_counts=True)

        # Plot histograms for Low Severity
        plt.figure(figsize=(10, 6))
        plt.plot(low_values, low_counts, color=color_palette[-2], marker='o', linestyle='-')
        plt.xlabel('AC_AF Values')
        plt.ylabel('Number of variants')
        plt.title('AC_AF Histogram for Pathogenic variants tagged with Severity Level: Low')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.savefig(save_path.replace('.png', '_low.png'))
        plt.close()

        # Plot histograms for High Severity
        plt.figure(figsize=(10, 6))
        plt.plot(high_values, high_counts, color=color_palette[2], marker='o', linestyle='-')
        plt.xlabel('AC_AF Values')
        plt.ylabel('Number of variants')
        plt.title('AC_AF Histogram for Benign variants tagged with Severity Level: High')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.savefig(save_path.replace('.png', '_high.png'))
        plt.close()

        print(f'Finished creating AC_AF histograms, results can be found at {save_path}')

    def adaptive_binning(data, max_bins, max_val):
        """
        Dynamically determine bin edges based on the density of data points.

        :param data: 1D array-like data
        :param max_bins: Maximum number of bins allowed
        :param max_val: Maximum value of the data
        :return: Array containing bin edges
        """
        # Calculate bin edges based on data density
        hist, bin_edges = np.histogram(data, bins='auto')
        bin_size = min(5, np.diff(bin_edges).min())  # Minimum bin size is 5
        if len(bin_edges) > max_bins:
            # If the number of bins exceeds the maximum allowed, merge adjacent bins to reduce the number
            bin_indices = np.argsort(hist)[::-1][:max_bins - 1]
            new_bin_edges = sorted([bin_edges[i] for i in bin_indices])
            new_bin_edges.insert(0, bin_edges[0])
            new_bin_edges.append(bin_edges[-1])
        else:
            new_bin_edges = bin_edges

        # Adjust bin edges to end at the actual data values
        sorted_data = np.sort(data)
        for i in range(1, len(new_bin_edges)):
            bin_edge = new_bin_edges[i]
            closest_index = np.searchsorted(sorted_data, bin_edge, side="right")
            new_bin_edges[i] = sorted_data[closest_index - 1]

        # Check if the last bin's endpoint is less than the maximum value
        if new_bin_edges[-1] < max_val:
            new_bin_edges.append(max_val)

        return np.array(new_bin_edges)

    def zoom_histogram(data, save_path):
        """
        Create a zoomed-in histogram plot focusing on each bin separately.

        :param data: 1D array-like data
        :param bin_counts: Array containing the counts of each bin
        :param bin_edges: Array containing the bin edges
        :param save_path: Path to save the histogram plot
        :return: None
        """
        bin_size = 1  # Adjust bin size as needed

        # Calculate the number of bins based on the bin_edges array
        num_bins = 50

        # Update the histogram
        plt.figure(figsize=(10, 6))
        # Create the histogram
        counts, _, _ = plt.hist(data, bins=np.arange(0, num_bins + 1), alpha=0.7, color=color_palette[-2],
                                edgecolor='black')
        plt.xlabel('AC_Hom Values', labelpad=-2)
        plt.ylabel('Number of Variants')
        plt.title('Zoomed-In High severity AC_Hom Histogram')
        plt.grid(True)
        plt.xticks(np.arange(0, num_bins + 1), rotation=45, ha='right', fontsize=8)  # Add ticks for each bin
        plt.xlim(-1, num_bins)  # Set the x-axis limit to zoom in to the desired scale

        # Add numbers on top of each bar representing the actual count
        for edge, count in enumerate(counts):
            plt.text(edge + 0.5, count, str(int(count)), ha='center', va='bottom', fontsize=8, rotation=45,
                     color='black')

        plt.savefig(save_path)
        plt.show()
        plt.close()

    def ac_hom(cv_path, save_path):
        """
        Generate histograms of the AC_Hom field values from the input file for high and low values separately
        and save them to the specified path.

        :param cv_path: Path to the input CSV file containing variant data
        :param save_path: Path to save the histogram plots
        :return: None
        """
        # Read the CSV file into a DataFrame
        print("Reading CSV file...")
        df = pd.read_csv(cv_path, sep='\t')

        # Check if the required columns are present in the DataFrame
        required_columns = ['AC_Hom', 'Severity']
        if not set(required_columns).issubset(df.columns):
            missing_columns = set(required_columns) - set(df.columns)
            raise KeyError(f"Missing columns in the CSV file: {missing_columns}")

        print("Filtering data by severity levels...")
        # Filter rows by severity levels Low and High
        low_severity = df[df['Severity'] == 'Low']
        high_severity = df[df['Severity'] == 'High']

        # Convert the 'AC_Hom' column to numeric, coercing errors to NaN
        low_ac_hom = pd.to_numeric(low_severity['AC_Hom'], errors='coerce')
        high_ac_hom = pd.to_numeric(high_severity['AC_Hom'], errors='coerce')

        # Round AC_Hom values to 4 decimal places
        low_ac_hom = low_ac_hom.round(4)
        high_ac_hom = high_ac_hom.round(4)

        # Get unique AC_Hom values and their counts for low and high severity
        low_values, low_counts = np.unique(low_ac_hom, return_counts=True)
        high_values, high_counts = np.unique(high_ac_hom, return_counts=True)

        # Plot histograms for Low Severity
        plt.figure(figsize=(10, 6))
        plt.plot(low_values, low_counts, color=color_palette[-2], marker='o', linestyle='-')
        plt.xlabel('AC_Hom Values')
        plt.ylabel('Number of Variants')
        plt.title('AC_Hom Histogram for Pathogenic variants tagged with Severity Level: Low')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # # Write count on top of each dot if the distance to the next dot is at least 0.5 away in the x-axis
        # for i in range(len(low_values) - 1):
        #     if low_values[i + 1] - low_values[i] >= 0.5:
        #         plt.text(low_values[i], low_counts[i], str(low_counts[i]), ha='center', va='bottom', fontsize=8)
        #
        # # Include the last dot
        # plt.text(low_values[-1], low_counts[-1], str(low_counts[-1]), ha='center', va='bottom', fontsize=8)

        plt.savefig(save_path.replace('.png', '_low.png'))
        plt.close()

        # Plot histograms for High Severity
        plt.figure(figsize=(10, 6))
        plt.plot(high_values, high_counts, color=color_palette[2], marker='o', linestyle='-')
        plt.xlabel('AC_Hom Values')
        plt.ylabel('Number of Variants')
        plt.title('AC_Hom Histogram for Benign variants tagged with Severity Level: High')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # # Write count on top of each dot if the distance to the next dot is at least 0.5 away in the x-axis
        # for i in range(len(high_counts) - 1):
        #     if high_values[i + 1] - high_values[i] >= 0.5:
        #         plt.text(high_values[i], high_counts[i], str(high_counts[i]), ha='center', va='bottom', fontsize=8)
        #
        # # Include the last dot
        # plt.text(high_values[-1], high_counts[-1], str(high_counts[-1]), ha='center', va='bottom', fontsize=8)

        plt.savefig(save_path.replace('.png', '_high.png'))
        plt.close()
        zoom_histogram(high_ac_hom, 'zoomin_' + save_path)
        print(f'Finished creating AC_Hom histograms, results can be found at {save_path}')

    def dftovcf(dataframe):
        vcf_path = 'lof.vcf'

        # Mapping between DataFrame columns and VCF columns
        column_mapping = {
            'Chromosome': 'CHROM',
            'PositionStart': 'POS',
            'RsNumber': 'ID',
            'REF': 'REF',
            'ALT': 'ALT'
        }

        # Create a new DataFrame with selected columns
        selected_columns = ['Chromosome', 'PositionStart', 'RsNumber', 'REF', 'ALT']
        vcf_df = dataframe[selected_columns].copy()

        # Rename columns based on the mapping
        vcf_df.rename(columns=column_mapping, inplace=True)

        # Add QUAL, FILTER, INFO, FORMAT columns with '.'
        vcf_df['QUAL'] = '.'
        vcf_df['FILTER'] = '.'
        vcf_df['INFO'] = '.'
        vcf_df['FORMAT'] = '.'

        # Replace empty values in 'RsNumber' column with '.'
        vcf_df['ID'] = vcf_df['ID'].replace('', '.')

        # Write DataFrame to VCF file
        vcf_df.to_csv(vcf_path, sep='\t', index=False)

        print(f"VCF file created: {vcf_path}")
        return vcf_path

    # Define paths and variables
    cv_tsv = 'cv_unexpected_variants_severity.txt'
    updated_cv_tsv_path = 'cv_unexpected_variants_severity_with_CLNREVSTAT.txt'
    revstatus_path = 'clnrevstat.png'
    af_histogram_path = 'af_histrogram.png'
    ac_hom_path = 'ac_hom.png'
    db = 'clinvar.vcf'
    analyze_path = 'analyze_output.xlsx'

    # Functions
    find_variants_cv_info(cv_tsv, db)
    # add_severity_condition(cv_tsv)
    clvrevstat_distribution_graph(updated_cv_tsv_path, revstatus_path)
    af_histogram(updated_cv_tsv_path, af_histogram_path)
    ac_hom(updated_cv_tsv_path, ac_hom_path)


def export_log_app(log_app):
    try:
        # Open a file for writing
        with open("program_export.txt", "w") as export_file:
            # Get the text from the text widget
            log_text = log_app.text_widget.get("1.0", tk.END)

            # Write the log text to the file
            export_file.write(log_text)

        # Signal the main application to exit
        sys.exit()
    except Exception as e:
        raise RuntimeError(f"Error exporting log data: {e}")


def main():
    """
    Main function for the Log App.
    Initiated once Run Program has been activated.
    """
    # Define file names
    # qa_gnomAD_file_path = "modified_sv_gnomad1.vcf.gz-Snv.txt"
    # gnomAD_excel = 'qa_gnomAD.xlsx'
    qa_clinvar_file_path = "clinvar_output_sv.vcf.gz-Snv.txt"
    clinvar_tsv = 'qa_clinvar.txt'
    clinvar_subgroupsizes_path = 'subgroups_sizes.xlsx'

    cv_unexpected_path = 'cv_unexpected_variants_severity.txt'
    cv_expected_path = 'cv_expected_variants_severity.txt'

    refiltered_file_path = 'qa_clinvar_refiltered.txt'

    # A file for testing
    test_file = 'test.txt'

    # Program functions :
    # Curate the snv, keep variants with ManeStatus = 1 or 2
    curate_result(qa_clinvar_file_path, clinvar_tsv)

    # Map filter encodings
    mapping_filter_encodings_qa(clinvar_tsv, refiltered_file_path)

    # Show subgroups sizes
    subgroups_length(refiltered_file_path, clinvar_subgroupsizes_path)

    # Divide to expected and unexpected variants
    expectancy_division(refiltered_file_path, cv_unexpected_path, cv_expected_path)

    # Collect raw statistics data
    total_length, expected_length, unexpected_length, hi_statistics, effect_statistics, cln_benign, cln_pathogenic, filters_statistics, auroc = collect_raw_data(
        cv_unexpected_path, cv_expected_path)

    # Visuallize collected data
    visualize_data(total_length, expected_length, unexpected_length, hi_statistics, effect_statistics, cln_benign,
                   cln_pathogenic, filters_statistics, auroc, refiltered_file_path)

    alphamissense_visualize(cv_unexpected_path, 'alphamissense_unexpected_spread.png')
    alphamissense_visualize(cv_expected_path, 'alphamissense_expected_spread.png')
    impact_visualize(cv_unexpected_path, 'impact_pie.png')

    # Study unexpected set
    # Test functions
    # Test effect mapping correctness
    # test_determine_effect_qa()

    # Test for subgroups containing unexpected variants at all by splitting the file according to each subgroup
    # split_file_by_filter(refiltered_file_path)


if __name__ == "__main__":
    # initialize the log app
    root = tk.Tk()
    log_app = InteractiveLog(root)
    # Run the log app
    root.mainloop()

    # Reset the standard output after the application closes
    sys.stdout = log_app.original_stdout
