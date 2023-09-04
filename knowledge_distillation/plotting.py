import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

# number_of_pairs = 9006
# split_size = number_of_pairs / 2

# en_test_directory = "EVMed/en/test"
# vi_test__directory = "EVMed/vi/test"
# kernel_graph_folder = "plot_pairs"

# os.makedirs(kernel_graph_folder, exist_ok=False)


# output_postive_test = os.path.join(kernel_graph_folder, "EVMed-{}-{}-pos-test.txt".format('en', 'vi'))
# output_negative_test = os.path.join(kernel_graph_folder, "EVMed-{}-{}-neg-test.txt".format('en', 'vi'))


# count = 0
# with open(output_postive_test, "a+") as fileOut:
#     for filename in sorted(os.listdir(en_test_directory)):
#         en_path = os.path.join(en_test_directory, filename)
#         vi_file_name = filename.replace("en","vi")
#         vi_path = os.path.join(vi_test__directory, vi_file_name)

#         with open(en_path, 'r', encoding='utf-8') as en_f, open(vi_path, 'r', encoding='utf-8') as vi_f:
#             en_lines = en_f.readlines()
#             vi_lines = vi_f.readlines()

#             if count < split_size:
#                 for en_line, vi_line in zip(en_lines, vi_lines):
#                     en_line = en_line.strip()
#                     vi_line = vi_line.strip()

#                     # Write the aligned pair with tab separation
#                     fileOut.write(f'{en_line}\t{vi_line}\n')
#                     count += 1
#             else:
#                 with open(output_negative_test, "a+") as neg_file:
#                     while True:
#                         shuffled_vi = deepcopy(vi_lines)

#                         random.shuffle(shuffled_vi)

#                         equivalent_pairs_exist = any(x == y for x, y in zip(vi_lines, shuffled_vi))

#                         if not equivalent_pairs_exist:
#                             break

#                     for en_line, vi_line in zip(en_lines, shuffled_vi):
#                         en_line = en_line.strip()
#                         vi_line = vi_line.strip()

#                         neg_file.write(f'{en_line}\t{vi_line}\n')
#                         count += 1


# ---------- PLOT FINETUNE -------------------------
ft_model = SentenceTransformer("output/make-multilingual-sys-2023-08-22_21-08-11")
ft_pos_embeddings = []
ft_neg_embeddings = []

with open("plot_pairs/EVMed-en-vi-pos-test.txt", "r") as pos_file:
    for idx, line in enumerate(pos_file):
        sentences = line.strip().split("\t")
        en_encode = ft_model.encode(sentences[0])
        vi_encode = ft_model.encode(sentences[-1])
        sim = util.cos_sim(en_encode, vi_encode).data.tolist()[0][0]
        ft_pos_embeddings.append(sim)

        if idx % 50 == 0:
            print(idx, ft_pos_embeddings[-1])

with open("plot_pairs/EVMed-en-vi-neg-test.txt", "r") as neg_file:
    for idx, line in enumerate(neg_file):
        sentences = line.strip().split("\t")
        en_encode = ft_model.encode(sentences[0])
        vi_encode = ft_model.encode(sentences[-1])
        sim = util.cos_sim(en_encode, vi_encode).data.tolist()[0][0]
        ft_neg_embeddings.append(sim)

        if idx % 50 == 0:
            print(idx, ft_neg_embeddings[-1])

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(
    data=ft_pos_embeddings,
    color="darkgreen",
    label="finetuned-postive-pairs",
    fill=True,
    ax=ax,
)
sns.kdeplot(
    data=ft_neg_embeddings,
    color="firebrick",
    label="finetuned-negative-pairs",
    fill=True,
    ax=ax,
)

# ---------- PLOT OG ---------------
og_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
og_pos_embeddings = []
og_neg_embeddings = []

with open("plot_pairs/EVMed-en-vi-pos-test.txt", "r") as pos_file:
    for idx, line in enumerate(pos_file):
        sentences = line.strip().split("\t")
        en_encode = og_model.encode(sentences[0])
        vi_encode = og_model.encode(sentences[-1])
        sim = util.cos_sim(en_encode, vi_encode).data.tolist()[0][0]
        og_pos_embeddings.append(sim)

        if idx % 50 == 0:
            print(idx, og_pos_embeddings[-1])

with open("plot_pairs/EVMed-en-vi-neg-test.txt", "r") as neg_file:
    for idx, line in enumerate(neg_file):
        sentences = line.strip().split("\t")
        en_encode = og_model.encode(sentences[0])
        vi_encode = og_model.encode(sentences[-1])
        sim = util.cos_sim(en_encode, vi_encode).data.tolist()[0][0]
        og_neg_embeddings.append(sim)

        if idx % 50 == 0:
            print(idx, og_neg_embeddings[-1])

sns.kdeplot(
    data=og_pos_embeddings, color="cyan", label="origin-postive-pairs", fill=True, ax=ax
)
sns.kdeplot(
    data=og_neg_embeddings,
    color="yellow",
    label="origin-negative-pairs",
    fill=True,
    ax=ax,
)

ax.legend()
plt.xlabel("Similarity")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
plt.savefig("combine-output.png")
