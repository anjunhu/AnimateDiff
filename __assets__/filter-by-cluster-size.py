"""
Filter clusters-10-500.json by:
  --min_size / --max_size       : cluster prompt count
  --min_tokens / --max_tokens   : per-prompt token length (CLIP tokenizer, averaged across cluster)
  --include_keywords            : keep cluster if ANY prompt matches ANY keyword
  --exclude_keywords            : drop cluster if ANY prompt matches ANY keyword
  Inclusion overrides exclusion: if a cluster satisfies an include keyword it is kept
  regardless of exclude keywords.

Output: filtered JSON in the same format.
"""

import json
import argparse
from transformers import CLIPTokenizer

def count_tokens(text, tokenizer):
    return tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].shape[1]

def matches_any(prompts, keywords):
    kw_lower = [k.lower() for k in keywords]
    return any(any(k in p.lower() for k in kw_lower) for p in prompts)

def avg_tokens(prompts, tokenizer):
    return sum(count_tokens(p, tokenizer) for p in prompts) / len(prompts)

def filter_clusters(clusters, args, tokenizer):
    result = {}
    stats = {"total": len(clusters), "kept": 0, "dropped_size": 0,
             "dropped_tokens": 0, "dropped_exclude": 0}

    for cluster_id, prompts in clusters.items():
        n = len(prompts)

        # --- size filter ---
        if args.min_size is not None and n < args.min_size:
            stats["dropped_size"] += 1
            continue
        if args.max_size is not None and n > args.max_size:
            stats["dropped_size"] += 1
            continue

        # --- token length filter ---
        if args.min_tokens is not None or args.max_tokens is not None:
            avg = avg_tokens(prompts, tokenizer)
            if args.min_tokens is not None and avg < args.min_tokens:
                stats["dropped_tokens"] += 1
                continue
            if args.max_tokens is not None and avg > args.max_tokens:
                stats["dropped_tokens"] += 1
                continue

        # --- keyword filter ---
        # inclusion overrides exclusion
        included = args.include_keywords and matches_any(prompts, args.include_keywords)
        excluded = args.exclude_keywords and matches_any(prompts, args.exclude_keywords)

        if excluded and not included:
            stats["dropped_exclude"] += 1
            continue

        result[cluster_id] = prompts
        stats["kept"] += 1

    return result, stats

'''
python filter-by-cluster-size.py --input  clusters-final.json  --min_size 50 --max_size 500 --min_tokens 10 --max_tokens 50 --exclude_keywords "abstract" "fire" "kaleidoscope" "aerial" --include_keywords "businessman" "doctor" "hologram" "flag" "football" "soccer" "airplane take off" "airplane landing" --output clusters-filtered.json 
'''

def main():
    parser = argparse.ArgumentParser(description="Filter clusters JSON")
    parser.add_argument("--input",  default="clusters-10-500.json")
    parser.add_argument("--output", default="clusters-filtered.json")
    parser.add_argument("--min_size",   type=int,   default=None, help="Min prompts per cluster")
    parser.add_argument("--max_size",   type=int,   default=None, help="Max prompts per cluster")
    parser.add_argument("--min_tokens", type=float, default=None, help="Min avg CLIP tokens per cluster")
    parser.add_argument("--max_tokens", type=float, default=None, help="Max avg CLIP tokens per cluster")
    parser.add_argument("--include_keywords", nargs="+", default=None,
                        help="Keep cluster if any prompt contains any of these keywords (overrides exclusion)")
    parser.add_argument("--exclude_keywords", nargs="+", default=None,
                        help="Drop cluster if any prompt contains any of these keywords")
    args = parser.parse_args()

    with open(args.input) as f:
        clusters = json.load(f)

    tokenizer = None
    if args.min_tokens is not None or args.max_tokens is not None:
        print("Loading CLIP tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    filtered, stats = filter_clusters(clusters, args, tokenizer)

    with open(args.output, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Total:            {stats['total']}")
    print(f"Kept:             {stats['kept']}")
    print(f"Dropped (size):   {stats['dropped_size']}")
    print(f"Dropped (tokens): {stats['dropped_tokens']}")
    print(f"Dropped (kw):     {stats['dropped_exclude']}")
    print(f"Output:           {args.output}")


if __name__ == "__main__":
    main()
