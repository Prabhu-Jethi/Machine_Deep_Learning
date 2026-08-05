from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
res = tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="tf",

    ### Padding ->
    ## Padding appends special tokens so shorter sequences match the longest sequence in a batch. 
    ## The attention mask marks padding positions as 0 so the model ignores them. 
    ## Set padding=True to pad to the longest sequence or pass max_length to pad to a fixed size.

    padding=True,

    ### Truncation ->
    ## Truncation clips tokens so a sequence fits within a maximum length. Set truncation=True and specify max_length to enable it.
    ## Padding and truncation work together. Short sequences gain padding tokens while long sequences lose trailing tokens.
    ## Together, they produce a packed rectangular tensor.

    truncation=True,
    max_length=5
)

print(res)


'''{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265],
        [     0,      2,   6519,    970,   3741,    675,   4105,  25955,  42184, 225789, 235265],
        [     0,      2,   2299,  73378,  17844,   4320, 224463,   4949,  48977,  9902, 235341]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}'''

'''{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656],
        [     2,   6519,    970,   3741,    675],
        [     2,   2299,  73378,  17844,   4320]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
}'''


