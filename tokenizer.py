from tqdm import tqdm
from kiwipiepy import Kiwi
import re


class tokenizer:
    def __init__(self, data_json):
        self.data_json = data_json
        self.date = []
        for i in self.data_json:
            self.date.append(i['date'])

    def make_sentence_pieces(self):
        doc = []
        for i in self.data_json:
            doc.append(i['title'] + " " + i['content'])
        preprocessed_documents = []
        for line in tqdm(doc):
            if line and not line.replace(' ', '').isdecimal():
                preprocessed_documents.append(line)

        return preprocessed_documents

    def init_tokenizer(self):
        tagger = Kiwi()
        sent = self.make_sentence_pieces()
        stop_words = ["EC", "EF", "VX", "SF", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "J", "SN", "X",
                      "SH", "SN", "SL", "SC", "SY", "SSO", "SSC", "SE", "SF", "V"]
        final_result = []
        for i in sent:
            word_tokens = tagger.tokenize(i)
            result = [word for word in word_tokens if len(word[0]) > 1 and word[0] not in ['본사', '기자']]
            packet = []
            for word in result:
                flag = 0
                for sw in stop_words:
                    ju = re.search(sw, word[1])
                    if ju:
                        flag = 1
                        break
                if flag == 0:
                    packet.append((word[0]))
            final_result.append(packet)
        return final_result
