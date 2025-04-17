import json

def filter_clusters_by_length(input_path, output_path, min_len=10, max_len=500):
    """
    Filters clusters in a JSON file based on their length and saves the result to a new file.

    Parameters:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
        min_len (int): Minimum cluster length to keep (inclusive).
        max_len (int): Maximum cluster length to keep (inclusive).
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = {
        k: v for k, v in data.items()
        if min_len <= len(v) <= max_len
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered {len(filtered_data)} clusters saved to '{output_path}'")

# Example usage:
filter_clusters_by_length('clusters-final.json', 'clusters-10-500.json', min_len=10, max_len=500)
