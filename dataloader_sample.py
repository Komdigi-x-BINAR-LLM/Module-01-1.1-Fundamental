import os
from torch.utils.data import Dataset, DataLoader

# struktur folder:
# data/
# ├── sample1/
# │   ├── question.txt
# │   └── answer.txt
# ├── sample2/
# │   ├── question.txt
# │   └── answer.txt
# ├── sample3/
# │   ├── question.txt
# │   └── answer.txt
# └── ...


class QADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        with open(
            os.path.join(sample_dir, "question.txt"), "r", encoding="utf-8"
        ) as fq:
            question = fq.read().strip()
        with open(os.path.join(sample_dir, "answer.txt"), "r", encoding="utf-8") as fa:
            answer = fa.read().strip()

        sample = {"question": question, "answer": answer}

        # misal proses tokenisasi atau sejenisnya
        if self.transform:
            sample = self.transform(sample)
        return sample

dataset = QADataset("data/")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

for batch in loader:
    print(batch["question"])
    print(batch["answer"])
    break
# Output
# ['Indonesia terdiri dari berapa provinsi?', 'Apa nama ibukota Indonesia?']
# ['Per tahun 2025, Indonesia terdiri dari 38 Provinsi.', 'Jakarta']

