
# Beat Tracking Error Detection

## Create beats

To create the fake beat predictions based on ground truth beat locations, run the following command:

```bash
python create_beats.py
```

This will generate the fake beat predictions in the `beats_with_error` directory.

## Create beat activations

To create the beat activations for audio files, run the following command:

```bash
python madmom_beat_acts.py
```

This will generate the beat activations for the audio files in the `beat_acts` directory.

## Call GPT

To call the GPT model with the generated prompts, run the following command:

```bash
python call_gpt.py
```

## Evaluate GPT

To evaluate the GPT model on detecting incorrect beat locations, run the following command:

```bash
python eval_gpt.py
```

This will output the evaluation results for the GPT model.