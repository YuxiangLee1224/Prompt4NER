from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange
import json


def read_data(args):
    train_data, dev_data, test_data = None, None, None
    if args.do_train:
        with open(args.train_path + args.train_file + '.json', 'r') as f:
            train_data = json.load(f)
    if args.do_dev:
        with open(args.dev_path + args.dev_file + '.json', 'r') as f:
            dev_data = json.load(f)
    if args.do_test:
        with open(args.test_path + args.test_file + '.json', 'r') as f:
            test_data = json.load(f)

    return train_data, dev_data, test_data


class TrainDataset(Dataset):
    def __init__(self, train_data, tokenizer):

        self.tokenizer = tokenizer
        self.encoded_prompt_ins = []
        self.encoded_prompt_outs = []
        self.slot_poses = []
        self.encoded_raw_sent = []

        for sidx in trange(len(train_data), desc='Loading train data'):

            prompt_tokens_ins = train_data[sidx]['prompt_seq_in']
            prompt_tokens_outs = train_data[sidx]['prompt_seq_out']
            original_tokens = train_data[sidx]['original_seq_in']

            original_tokens.append('none')
            original_tokens.append('@')
            original_tokens.append('.')

            encoded_raw_sent = tokenizer.encode(" ".join(original_tokens), add_prefix_space=True)
            encoded_raw_sent.append(tokenizer.eos_token_id)
            onesent_encoded_prompt_in = []
            onesent_encoded_prompt_out = []
            onesent_slot_poses = []
            onesent_encoded_raw_sent = []
            for lidx in range(len(prompt_tokens_ins)):
                prompt_tokens_in = prompt_tokens_ins[lidx]
                prompt_tokens_out = prompt_tokens_outs[lidx]
                slots = prompt_tokens_out[len(prompt_tokens_in):]
                encoded_prompt_in = tokenizer.encode(" ".join(prompt_tokens_in).replace("'", '"'),
                                                     add_prefix_space=True)
                encoded_slots = tokenizer.encode(" ".join(slots) + " .<|endoftext|>", add_prefix_space=True)

                encoded_prompt_out = encoded_prompt_in + encoded_slots
                slot_pos = [-100] * len(encoded_prompt_in)

                for i in encoded_slots:
                    slot_pos.append(i)
                onesent_slot_poses.append(slot_pos)
                onesent_encoded_raw_sent.append(encoded_raw_sent)
                onesent_encoded_prompt_in.append(encoded_prompt_in)
                onesent_encoded_prompt_out.append(encoded_prompt_out)
            self.encoded_prompt_ins.extend(onesent_encoded_prompt_in)
            self.encoded_prompt_outs.extend(onesent_encoded_prompt_out)
            self.encoded_raw_sent.extend(onesent_encoded_raw_sent)
            self.slot_poses.extend(onesent_slot_poses)


    def __getitem__(self, index):
        return self.encoded_prompt_ins[index], self.encoded_prompt_outs[index], self.encoded_raw_sent[index], \
               self.slot_poses[index]

    def __len__(self):
        return len(self.encoded_prompt_ins)



def traincollate(data):
    out_sen_len = 0
    for j in data:
        if len(j[1]) > out_sen_len:
            out_sen_len = len(j[1])
    in_sen_len = 0
    for j in data:
        if len(j[0]) > in_sen_len:
            in_sen_len = len(j[0])

    batch_encoded_prompt_in = []
    batch_encoded_prompt_out = []
    batch_mask_encoded_prompt_out = []
    batch_slot_poses = []
    batch_encoded_raw_sent = []
    batch_attention_mask = []

    for i in data:

        encoded_prompt_in = i[0]
        encoded_prompt_in = encoded_prompt_in + (in_sen_len - len(encoded_prompt_in)) * [50256]
        encoded_prompt_out = i[1]
        mask_encoded_prompt_out = i[1]
        attention_mask = len(encoded_prompt_out) * [1] + (out_sen_len - len(encoded_prompt_out)) * [0]
        slot_pos = i[3]
        slot_pos = slot_pos + (out_sen_len - len(slot_pos)) * [-100]
        encoded_prompt_out = encoded_prompt_out + (out_sen_len - len(encoded_prompt_out)) * [50256]
        mask_encoded_prompt_out = mask_encoded_prompt_out + (out_sen_len - len(mask_encoded_prompt_out)) * [-100]
        encoded_raw_sent = i[2]
        batch_encoded_prompt_in.append(encoded_prompt_in)
        batch_encoded_prompt_out.append(encoded_prompt_out)
        batch_mask_encoded_prompt_out.append(mask_encoded_prompt_out)
        batch_encoded_raw_sent.append(encoded_raw_sent)
        batch_attention_mask.append(attention_mask)
        batch_slot_poses.append(slot_pos)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_encoded_prompt_in = torch.tensor(batch_encoded_prompt_in, dtype=torch.long, device=device)
    batch_encoded_prompt_out = torch.tensor(batch_encoded_prompt_out, dtype=torch.long, device=device)
    batch_mask_encoded_prompt_out = torch.tensor(batch_mask_encoded_prompt_out, dtype=torch.long, device=device)
    batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)
    batch_slot_poses = torch.tensor(batch_slot_poses, dtype=torch.long, device=device)

    return (batch_encoded_prompt_in, batch_encoded_prompt_out, batch_mask_encoded_prompt_out, batch_encoded_raw_sent,
            batch_attention_mask, batch_slot_poses)


class DevDataset(Dataset):
    def __init__(self, dev_data, tokenizer):
        self.tokenizer = tokenizer
        self.support_encoded_prompt_in = []
        self.support_encoded_prompt_out = []
        self.support_encoded_raw_sent = []
        self.support_mask_encoded_prompt_out = []

        self.test_encoded_prompt_in = []
        self.test_encoded_prompt_out = []
        self.test_encoded_raw_sent = []
        self.test_mask_encoded_prompt_out = []

        for domain_name in dev_data.keys():
            for eid, episode in enumerate(dev_data[domain_name]):
                support_prompt_in = episode['support']['prompt_seq_in']
                support_prompt_out = episode['support']['prompt_seq_out']
                encoded_support_prompt_in = [
                    tokenizer(" ".join(sent_label).replace("'", '"'), add_prefix_space=True)['input_ids'] for sent in
                    support_prompt_in for sent_label in sent]
                encoded_support_prompt_out = [
                    tokenizer(" ".join(sent_label).replace("'", '"') + " .<|endoftext|>", add_prefix_space=True)[
                        'input_ids'] for sent in support_prompt_out for sent_label in sent]

                encoded_support_mask_prompt_out = [len(sent) * [-100] + encoded_support_prompt_out[sid][len(sent):] for
                                                   sid, sent in enumerate(encoded_support_prompt_in)]

                self.support_encoded_prompt_in.append(encoded_support_prompt_in)
                self.support_encoded_prompt_out.append(encoded_support_prompt_out)
                self.support_mask_encoded_prompt_out.append(encoded_support_mask_prompt_out)

                test_prompt_in = episode['test']['prompt_seq_in']
                test_prompt_out = episode['test']['prompt_seq_out']
                encoded_test_prompt_in = [
                    tokenizer(" ".join(sent_label).replace("'", '"'), add_prefix_space=True)['input_ids'] for sent in
                    test_prompt_in for sent_label in sent]
                encoded_test_prompt_out = [
                    tokenizer(" ".join(sent_label).replace("'", '"') + " .<|endoftext|>", add_prefix_space=True)[
                        'input_ids'] for sent in test_prompt_out for sent_label in sent]

                encoded_test_mask_prompt_out = [len(sent) * [-100] + encoded_test_prompt_out[sid][len(sent):] for
                                                sid, sent in enumerate(encoded_test_prompt_in)]

                self.test_encoded_prompt_in.append(encoded_test_prompt_in)
                self.test_encoded_prompt_out.append(encoded_test_prompt_out)
                self.test_mask_encoded_prompt_out.append(encoded_test_mask_prompt_out)

                oneepisode_encoded_raw_sent_support = []
                oneepisode_encoded_raw_sent_test = []

                for sent_id, onesent_prompts in enumerate(episode['support']['prompt_seq_in']):

                    onesent_encoded_raw_sent = []
                    raw_sent = episode['support']['original_seq_in'][sent_id]
                    encoded_raw_sent = tokenizer.encode(" ".join(raw_sent), add_prefix_space=True)
                    encoded_raw_sent.append(2488)
                    encoded_raw_sent.append(4844)
                    encoded_raw_sent.append(764)
                    encoded_raw_sent.append(tokenizer.eos_token_id)
                    # len(onesent_prompts) == len(label)
                    for i in range(len(onesent_prompts)):
                        onesent_encoded_raw_sent.append(encoded_raw_sent)
                    oneepisode_encoded_raw_sent_support.extend(onesent_encoded_raw_sent)

                for sent_id, onesent_prompts in enumerate(episode['test']['prompt_seq_in']):
                    onesent_encoded_raw_sent = []
                    raw_sent = episode['test']['original_seq_in'][sent_id]
                    encoded_raw_sent = tokenizer.encode(" ".join(raw_sent), add_prefix_space=True)
                    encoded_raw_sent.append(2488)
                    encoded_raw_sent.append(4844)
                    encoded_raw_sent.append(764)
                    encoded_raw_sent.append(tokenizer.eos_token_id)
                    for i in range(len(onesent_prompts)):
                        onesent_encoded_raw_sent.append(encoded_raw_sent)
                    oneepisode_encoded_raw_sent_test.extend(onesent_encoded_raw_sent)

                self.support_encoded_raw_sent.append(oneepisode_encoded_raw_sent_support)
                self.test_encoded_raw_sent.append(oneepisode_encoded_raw_sent_test)

    def __len__(self):
        return len(self.support_encoded_prompt_in)

    def __getitem__(self, index):
        return self.support_encoded_prompt_in[index], self.support_encoded_prompt_out[index], \
               self.support_encoded_raw_sent[index], self.support_mask_encoded_prompt_out[index], \
               self.test_encoded_prompt_in[index], self.test_encoded_prompt_out[index], self.test_encoded_raw_sent[
                   index], self.test_mask_encoded_prompt_out[index]

class DevSupportEpisode(Dataset):
    def __init__(self, support_encoded_prompt_in, support_encoded_prompt_out, support_encoded_raw_sent,
                 support_mask_encoded_prompt_out, tokenizer):
        self.support_encoded_prompt_in = support_encoded_prompt_in
        self.support_encoded_prompt_out = support_encoded_prompt_out
        self.support_encoded_raw_sent = support_encoded_raw_sent
        self.support_mask_encoded_prompt_out = support_mask_encoded_prompt_out

    def __len__(self):
        return len(self.support_encoded_prompt_in)

    def __getitem__(self, index):
        return self.support_encoded_prompt_in[index], self.support_encoded_prompt_out[index], \
               self.support_encoded_raw_sent[index], self.support_mask_encoded_prompt_out[index]

class DevTestEpisode(Dataset):
    def __init__(self, test_encoded_prompt_in, test_encoded_prompt_out, test_encoded_raw_sent,
                 test_mask_encoded_prompt_out, tokenizer):
        self.test_encoded_prompt_in = test_encoded_prompt_in
        self.test_encoded_prompt_out = test_encoded_prompt_out
        self.test_encoded_raw_sent = test_encoded_raw_sent
        self.test_mask_encoded_prompt_out = test_mask_encoded_prompt_out

    def __len__(self):
        return len(self.test_encoded_prompt_in)

    def __getitem__(self, index):
        return self.test_encoded_prompt_in[index], self.test_encoded_prompt_out[index], self.test_encoded_raw_sent[
            index], self.test_mask_encoded_prompt_out[index]

def devcollate(data):
    out_sen_len = 0
    for i in data:
        if len(i[1]) > out_sen_len:
            out_sen_len = len(i[1])
    in_sen_len = 0
    for i in data:
        if len(i[0]) > in_sen_len:
            in_sen_len = len(i[0])
    raw_sen_len = 0
    for i in data:
        if len(i[2]) > raw_sen_len:
            raw_sen_len = len(i[2])
    batch_prompt_in = []
    batch_prompt_out = []
    batch_raw_sent = []
    batch_attention_mask = []
    batch_masked_prompt_out = []
    batch_in_length = []
    for item in data:

        prompt_in = []
        prompt_out = []
        masked_prompt_out = []
        attention_mask = []
        raw_sent = []
        batch_in_length.append(len(item[0]))
        prompt_in.append(item[0] + (in_sen_len - len(item[0])) * [50256])
        attention_mask.append(len(item[1]) * [1] + (out_sen_len - len(item[1])) * [0])
        prompt_out.append(item[1] + (out_sen_len - len(item[1])) * [50256])
        masked_prompt_out.append(item[3] + (out_sen_len - len(item[3])) * [-100])
        raw_sent.append(item[2] + (raw_sen_len - len(item[2])) * [50256])

        batch_prompt_in.extend(prompt_in)
        batch_prompt_out.extend(prompt_out)
        batch_masked_prompt_out.extend(masked_prompt_out)
        batch_attention_mask.extend(attention_mask)
        batch_raw_sent.extend(raw_sent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_prompt_in = torch.tensor(batch_prompt_in, dtype=torch.long, device=device)
    batch_prompt_out = torch.tensor(batch_prompt_out, dtype=torch.long, device=device)
    batch_masked_prompt_out = torch.tensor(batch_masked_prompt_out, dtype=torch.long, device=device)
    batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)

    return batch_prompt_in, batch_prompt_out, batch_masked_prompt_out, batch_attention_mask, batch_raw_sent, batch_in_length
