import torch
import clip
import collections
import pickle
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open('/root/code/GTnet/GTnet-master/dataset/Video/tee/84.jpg')).unsqueeze(0).to(device)
text = clip.tokenize(["sea","tee","wash_car"]).to(device)
class_0=["sea","tee","wash_car"]
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
print('1',image_features.shape)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
print('2',image_features.shape)
exit()
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(3)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{class_0[index]:>16s}: {100 * value.item():.2f}%")

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
# test save plk
with torch.no_grad():
    output_dict = collections.defaultdict(list)
    for value, index in zip(values, indices):
        output_dict[index.item()].append(value)
    all_info = output_dict
    save_pickle('/root/code/GTnet/GTnet-master/checkpoints/dataset/WideResNet28_10_S2M2_R/clip' + '/output.plk', all_info)
