# Annotation Tool

Contains normalized, EWTS, tshad-split lines for files from the
ACIP sungbum.

## Usage

NB: In this repo, I was concerned with the Nyingma collection. You'll need to basically
replace anything "Nyingma" with "Sungbum".

First, you'll need to use the "verse_01_detection_v2_nyingma.py". It takes as input a
directory that has all the Nyingma files in one directory. (The Sungbum is presented by
Asian Legacy Library using subfolders, so you'll just need to flatten those.)

As output, pick an empty dir. You'll get one json file per input file.

These output files contain spans of text that constitue a "verse", including the metadata
about that verse (file it came from, line numbers, confidence of this being a verse,
meter syllable count, and the verse lines itself).

This json format is the expected input for the next step.

### Using the json intermediate format

Now switch over to verse_03_deallo_nyingma.py. This script takes as input the directory
containing the json files created by the last script.

Choose any empty output dir.

You will need to modify this line:

```py
derge_corpus_path = "./derge_tshad_split_ewts"
```

This should point to wherever your Derge corpus lives.

The output will be a new json per input file.

Here, you'll see the number of exact matches for each Sungbum verse that was found in the
Derge, and the confidence of that match.

## Interpreting these results

You can now decide what you want to do. For each json file in the final output dir, you
can check each verse detected, see whether it was found in the Derge, and whether you
might want to eliminate it from further analysis.

I do not have code to perform these deletions, but it should be simple to modify the
second script to output a new Sungbum file for each json, with the given lines deleted, if
you want to remove that verse (depending on whatever metric you decide upon).

### Labeling for classification

If you want to label individual lines from a Sungbum document as either allochthonous
or authochthonous, you can use the final output files to determine if a given line number
was inside a Derge-matched quote or not.

## skwarelog

There's a custom logging library in use in these scripts. The simplest way around this is
to just replace the logging calls with print statements.

Beyond that, I think tqdm is the only dependency you'll need to install.
