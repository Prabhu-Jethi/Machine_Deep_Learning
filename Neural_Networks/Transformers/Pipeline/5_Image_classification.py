
from transformers import pipeline

pipe = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224"
)
result = pipe(
    inputs="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)


'''[{'label': 'lynx, catamount', 'score': 0.43350082635879517}, 
{'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'score': 0.034796182066202164}, 
{'label': 'snow leopard, ounce, Panthera uncia', 'score': 0.03240184113383293}, 
{'label': 'Egyptian cat', 'score': 0.023944802582263947}, 
{'label': 'tiger cat', 'score': 0.022889181971549988}]'''