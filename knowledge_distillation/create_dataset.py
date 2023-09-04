import os

source_lang = "en"  # Languages our (monolingual) teacher model understands
target_lang = "vi"  # New languages we want to extend to

dataset_folder = "EVMed"
parallel_sentences_folder = "EVMed-parallel-sentences/"
os.makedirs(parallel_sentences_folder, exist_ok=False)

output_filename_train = os.path.join(
    parallel_sentences_folder, "EVMed-{}-{}-train.txt".format(source_lang, target_lang)
)
output_filename_val = os.path.join(
    parallel_sentences_folder, "EVMed-{}-{}-val.txt".format(source_lang, target_lang)
)

# Create train
en_train_directory = "EVMed/en/train"
vi_train_directory = "EVMed/vi/train"


with open(output_filename_train, "a+") as fileOut:
    for filename in sorted(os.listdir(en_train_directory)):
        en_path = os.path.join(en_train_directory, filename)
        vi_file_name = filename.replace("en", "vi")
        vi_path = os.path.join(vi_train_directory, vi_file_name)

        with open(en_path, "r", encoding="utf-8") as en_f, open(
            vi_path, "r", encoding="utf-8"
        ) as vi_f:
            en_lines = en_f.readlines()
            vi_lines = vi_f.readlines()

            for en_line, vi_line in zip(en_lines, vi_lines):
                en_line = en_line.strip()
                vi_line = vi_line.strip()

                # Write the aligned pair with tab separation
                fileOut.write(f"{en_line}\t{vi_line}\n")

# Create val
en_val_directory = "EVMed/en/val"
vi_val_directory = "EVMed/vi/val"


with open(output_filename_val, "a+") as fileOut:
    for filename in sorted(os.listdir(en_val_directory)):
        en_path = os.path.join(en_val_directory, filename)
        vi_file_name = filename.replace("en", "vi")
        vi_path = os.path.join(vi_val_directory, vi_file_name)

        with open(en_path, "r", encoding="utf-8") as en_f, open(
            vi_path, "r", encoding="utf-8"
        ) as vi_f:
            en_lines = en_f.readlines()
            vi_lines = vi_f.readlines()

            for en_line, vi_line in zip(en_lines, vi_lines):
                en_line = en_line.strip()
                vi_line = vi_line.strip()

                # Write the aligned pair with tab separation
                fileOut.write(f"{en_line}\t{vi_line}\n")

print("---DONE---")
