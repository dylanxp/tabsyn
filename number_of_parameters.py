import torch
import sys
import os
import glob
import fnmatch
import json
import csv
import re

def count_tensors(obj):
    """Recursively count tensor elements in nested structures, excluding integer tensors."""
    if isinstance(obj, torch.Tensor):
        if obj.dtype.is_floating_point or obj.dtype == torch.bool or obj.dtype == torch.complex64 or obj.dtype == torch.complex128:
            return obj.numel()
        else:
            return 0
    elif isinstance(obj, dict):
        return sum(count_tensors(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(count_tensors(v) for v in obj)
    else:
        return 0

def count_parameters_in_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if "CTGAN" in checkpoint_path:
        total_params = 0
        generator_params = sum(p.numel() for p in checkpoint._generator.parameters() if p.requires_grad)
        total_params += generator_params
    elif "TVAE" in checkpoint_path:
        total_params = 0
        decoder_params = sum(p.numel() for p in checkpoint.decoder.parameters() if p.requires_grad)
        total_params += decoder_params
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)
        total_params = count_tensors(state_dict)
    return total_params

def scrape_external_parameter_count():
    """
    Scrapes the encoder or discriminator parameter counts from a folder of training log files.
    Returns a dictionary with all the counts.
    """
    methods = ["ctgan", "tvae"]
    datasets = ["adult", "beijing", "default", "diabetes", "magic", "news", "shoppers"]

    pattern = os.path.join("result_times", "*_*_out.txt")

    param_counts = {}

    print("Scraping external parameters (encoder/discriminator) from log files...")

    for filepath in glob.glob(pattern):
        match = re.search(r'([^/]+)/([^_]+)_([^_]+)_out\.txt$', filepath)
        if not match:
            print(f"Filename {filepath} does not match expected pattern.")
            continue
        method = match.group(2).lower()
        dataset = match.group(3).lower()
        # print("Method, dataset:", method, dataset )
        if method not in methods or dataset not in datasets:
            continue
        with open(filepath, "r") as f:
            lines = f.readlines()
            param_count = None
            for line in lines:
                m = re.search(r"Number of parameters =\s*(\d+)", line, re.IGNORECASE)
                if m:
                    param_count = int(m.group(1))
                    break
            # print(f'Method: {method}, Dataset: {dataset}, External Parameters: {param_count}')
            if param_count is None:
                continue  # or handle missing value

            # print(encoder_params, discriminator_params)
        key = (method, dataset)
        param_counts[key] = param_count
    
    return param_counts

def find_files(*, starting_folder: str = ".", pattern: str):
    """
    find all files that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full_name = os.path.join(root, filename)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches

def printc(message, color):
    """
    Print a message to the terminal in the specified color.

    color: one of "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    """
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }
    color_code = colors.get(color.lower(), colors["reset"])

    if isinstance(message, dict):
        message = json.dumps(message, indent=4)
    print(f"{color_code}{message}{colors['reset']}")

if __name__ == "__main__":    
    """
    Example usage:
    python number_of_parameters.py <checkpoint_path>
    OR
    python number_of_parameters.py <starting_folder> <pattern>
    OR
    python number_of_parameters.py <starting_folder>
    (in the last case, it will search for *.pt, *.pth, *.ckpt files)
    """

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            # If argument is a directory, use default patterns
            # patterns = ["*.pt", "*.ckpt", "*.pth"]
            patterns = ["*model.pt", "*model.pth", "*model_dis.pt", "*model_con.pt"]
            checkpoint_paths = []
            for pattern in patterns:
                checkpoint_paths.extend(find_files(starting_folder=arg, pattern=pattern))
        else:
            checkpoint_paths = [arg]
    elif len(sys.argv) == 3:
        checkpoint_paths = find_files(starting_folder=sys.argv[1], pattern=sys.argv[2])
    else:
        # print("Usage: python number_of_parameters.py <checkpoint_path> OR python number_of_parameters.py <starting_folder> <pattern>")
        # sys.exit(1)
        patterns = ["*model.pt", "*model.pth", "*model_dis.pt", "*model_con.pt", "*encoder.pt", "*decoder.pt"]
        checkpoint_paths = []
        for pattern in patterns:
            checkpoint_paths.extend(find_files(starting_folder="baselines/", pattern=pattern))
            checkpoint_paths.extend(find_files(starting_folder="tabsyn/", pattern=pattern))

    param_counts = scrape_external_parameter_count()
    # print(param_counts)

    methods_to_check = ["ctgan", "tvae", "codi", "stasy", "tabddpm", "vae", "tabsyn", "great"]
    
    print("Scraping model parameters from checkpoint files...")
    for checkpoint_path in checkpoint_paths:
        try:
            split_path = checkpoint_path.split(os.sep)
            ckpt_idx = split_path.index("ckpt")
            method = split_path[ckpt_idx - 1].lower()
            dataset = split_path[ckpt_idx + 1].lower()
            if method not in methods_to_check:
                continue
            num_params = count_parameters_in_checkpoint(checkpoint_path)
            # printc(f"{checkpoint_path}: {num_params}", "yellow")
            key = (method, dataset)
            if method in ["ctgan", "tvae"]:
                param_counts[key] += num_params
            if method in ["tabsyn", "vae"]:
                key = ("tabsyn", dataset)
                if key in param_counts.keys():
                    param_counts[key] += num_params
                else:
                    param_counts[key] = num_params
            else:
                param_counts[key] = num_params
        except Exception as e:
            printc(f"Error processing {checkpoint_path}: {e}", "red")
            continue

    # Sort the dictionary by method name
    param_counts = dict(sorted(param_counts.items(), key=lambda item: item[0]))

    # Write data to csv
    output_csv = "parameter_counts.csv"
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["method", "dataset", "num_parameters"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (method, dataset), num_params in param_counts.items():
            writer.writerow({
                "method": method,
                "dataset": dataset,
                "num_parameters": num_params
            })
    printc(f"Wrote parameter counts to {output_csv}", "green")            