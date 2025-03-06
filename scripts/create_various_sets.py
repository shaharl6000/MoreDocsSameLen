import json
import tiktoken
import random
import wikipedia
import torch
import os
from argparse import ArgumentParser
import openai
import copy
from tqdm import tqdm

""" -------------GLOBAL VARIABLES--------------- """

os.environ['CURL_CA_BUNDLE'] = ''
VOCAB_SIZE_TOKENIZER = 100256
wikipedia_cache = {}

""" -------------HELPER FUNCTIONS--------------- """


def count_tokens(text, encoding):
    return len(encoding.encode(text))


def add_random_tokens_end(paragraph_text, num_tokens_to_add, encoding):
    vocab_size = VOCAB_SIZE_TOKENIZER
    random_text = "".join(encoding.decode([random.randint(0, vocab_size - 1)]) for _ in range(num_tokens_to_add))
    return paragraph_text + random_text


def add_pad_tokens_end(paragraph_text, num_tokens_to_add, token_end):
    random_text = "".join(token_end for _ in range(num_tokens_to_add))
    return paragraph_text + random_text


def get_wikipedia_page_content(title):
    if title in wikipedia_cache:
        return wikipedia_cache[title]

    search_results = wikipedia.search(title)
    if not search_results:
        wikipedia_cache[title] = False
        return False
    try:
        page = wikipedia.page(search_results[0])
        wikipedia_cache[title] = page.content
        return page.content
    except wikipedia.exceptions.PageError:
        try:
            page = wikipedia.page(search_results[1])
            wikipedia_cache[title] = page.content
            return page.content
        except Exception as ex:
            wikipedia_cache[title] = False
            return False
    except wikipedia.exceptions.DisambiguationError as e:
        # print(f"Disambiguation page. Options: {e.options}")
        try:
            page = wikipedia.page(e.options[0])
            wikipedia_cache[title] = page.content
            return page.content
        except Exception as ex:
            wikipedia_cache[title] = False
            return False
    except wikipedia.exceptions.RedirectError as e:
        # print(f"Redirected page. New title: {e.title}")
        try:
            page = wikipedia.page(e.title)
            wikipedia_cache[title] = page.content
            return page.content
        except Exception as ex:
            wikipedia_cache[title] = False
            return False
    except Exception as e:
        wikipedia_cache[title] = False
        return False


def add_wikipedia_tokens_end(paragraph_text, paragraph_title, num_tokens_to_add, encoding):
    wiki_page = get_wikipedia_page_content(paragraph_title)
    wiki_page_encoded = encoding.encode(paragraph_text if wiki_page is False else wiki_page)
    tokens_to_add = (wiki_page_encoded * ((num_tokens_to_add // len(wiki_page_encoded)) + 1))[:num_tokens_to_add]
    added_content = "".join(encoding.decode([t]) for t in tokens_to_add)
    return paragraph_text + added_content


def add_wikipedia_tokens_wrap(paragraph_text, paragraph_title, num_tokens_to_add_before, num_tokens_to_add_after,
                              encoding):
    wiki_page = get_wikipedia_page_content(paragraph_title)
    wiki_page_encoded = encoding.encode(paragraph_text if wiki_page is False else wiki_page)

    # remove start of before cintent since it is frequently match the paragraph text
    paragraph_text_encoded_len = len(encoding.encode(paragraph_text))
    num_tokens_to_add_before += paragraph_text_encoded_len

    tokens_to_add_before = (wiki_page_encoded * ((num_tokens_to_add_before // len(wiki_page_encoded)) + 1))[
                           :num_tokens_to_add_before]
    tokens_to_add_after = (wiki_page_encoded * ((num_tokens_to_add_after // len(wiki_page_encoded)) + 1))[
                          :num_tokens_to_add_after]

    added_content_before = "".join(encoding.decode([t]) for t in tokens_to_add_before[paragraph_text_encoded_len:])
    added_content_after = "".join(encoding.decode([t]) for t in tokens_to_add_after)

    return added_content_before + paragraph_text + added_content_after


def truncate_or_pad_text(text, target_tokens, encoding, add_random=False):
    current_tokens = encoding.encode(text)
    current_length = len(current_tokens)
    if current_length > target_tokens:
        truncated_text = encoding.decode(current_tokens[:target_tokens])
        return truncated_text
    elif current_length < target_tokens:
        if add_random:
            num_extra_tokens = target_tokens - current_length
            return add_random_tokens_end(text, num_extra_tokens, encoding)
        else:
            return truncate_or_pad_text(text + text, target_tokens, encoding, add_random=False)
    else:
        return text
    

def create_rephrased_questions(input_path, output_path):
    demonstration = "Original question: What's #1 's hockey club named? " \
                    "Rephrased question: What is the hockey club called for the team ranked number one?"

    with open(input_path, 'r') as infile:
        total_lines = sum(1 for line in infile)

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for i, line in enumerate(tqdm(infile, total=total_lines, desc="Processing lines")):
            data = json.loads(line)
            original_question = data["question"]
            prompt = f"Given a following question, rephrase it to maintain the exact idea but change the phrasing as much as possible." \
                     f" For example: {demonstration} Rephrase the original question: {original_question} Rephrased question:"

            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            data["original_question"] = original_question

            data["question"] = completion.choices[0].message.content
            outfile.write(json.dumps(data) + '\n')


def rephrase_question_in_set(rephrased_path, input_path, output_path):
    with open(rephrased_path, 'r') as rffile, open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for rf_line, in_line in zip(rffile, infile):
            data = json.loads(in_line)
            reference = json.loads(rf_line)
            data["question"] = reference["question"]
            outfile.write(json.dumps(data) + '\n')


""" -------------CREATE SETS FUNCTIONS--------------- """


def create_original_collection(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            if data.get('answerable') is True:
                outfile.write(json.dumps(data) + '\n')


def create_oracle(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            supporting_paragraphs = [p for p in data.get('paragraphs', []) if p.get('is_supporting') is True]
            if supporting_paragraphs:
                # Replace the paragraphs with only the supporting ones
                data['paragraphs'] = supporting_paragraphs
                outfile.write(json.dumps(data) + '\n')


def create_no_questions(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            data['paragraphs'] = []
            outfile.write(json.dumps(data) + '\n')


def create_replace_distractors(input_path, output_path, num_to_retain=None):
    encoding = tiktoken.encoding_for_model("gpt-4")

    with open(input_path, 'r') as infile:
        lines = [json.loads(line) for line in infile]

    all_non_supporting_paragraphs = []
    for line in lines:
        non_supporting_paragraphs = [p for p in line.get('paragraphs', []) if not p.get('is_supporting')]
        all_non_supporting_paragraphs.extend(non_supporting_paragraphs)

    with open(output_path, 'w') as outfile:
        for line in lines:
            supporting_paragraphs = [p for p in line.get('paragraphs', []) if p.get('is_supporting')]
            non_supporting_paragraphs = [p for p in line.get('paragraphs', []) if not p.get('is_supporting')]

            if num_to_retain is not None and num_to_retain < len(non_supporting_paragraphs):
                retained_paragraphs = non_supporting_paragraphs[:num_to_retain]
                paragraphs_to_replace = non_supporting_paragraphs[num_to_retain:]
            else:
                retained_paragraphs = []
                paragraphs_to_replace = non_supporting_paragraphs

            for p in paragraphs_to_replace:
                target_tokens = count_tokens(p['paragraph_text'], encoding)
                new_paragraph = random.choice(all_non_supporting_paragraphs)
                new_paragraph_text = truncate_or_pad_text(new_paragraph['paragraph_text'], target_tokens, encoding)
                p['paragraph_text'] = new_paragraph_text

            data = {
                'id': line['id'],
                'paragraphs': supporting_paragraphs + retained_paragraphs + paragraphs_to_replace,
                'question': line['question'],
                'question_decomposition': line['question_decomposition'],
                'answer': line['answer'],
                'answer_aliases': line['answer_aliases'],
                'answerable': line['answerable']
            }

            outfile.write(json.dumps(data) + '\n')


def create_expanded(input_path, output_path, num_to_retain=None):
    encoding = tiktoken.encoding_for_model("gpt-4")

    with open(input_path, 'r') as infile:
        total_lines = sum(1 for line in infile)

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for i, line in enumerate(tqdm(infile, total=total_lines, desc="Processing lines")):
            data = json.loads(line)
            supporting_paragraphs = [p for p in data.get('paragraphs', []) if p.get('is_supporting') is True]
            non_supporting_paragraphs = [p for p in data.get('paragraphs', []) if p.get('is_supporting') is False]

            if num_to_retain is not None:
                cur_num_to_retain = min(num_to_retain - len(supporting_paragraphs), len(non_supporting_paragraphs))
                random_non_supporting = random.sample(non_supporting_paragraphs, cur_num_to_retain)
                supporting_paragraphs.extend(random_non_supporting)
                supporting_paragraphs.sort(key=lambda p: p['idx'])
                non_supporting_paragraphs = [p for p in non_supporting_paragraphs if p not in random_non_supporting]

            non_supporting_paragraphs_tokens = [[p['idx'], count_tokens(p['paragraph_text'], encoding)]
                                                for p in non_supporting_paragraphs]

            num_supporting_paragraphs = len(supporting_paragraphs)

            if num_supporting_paragraphs > 0:
                for i, p in enumerate(supporting_paragraphs):
                    if i == 0:
                        num_tokens_to_add_before = \
                            sum([tokens[1] for tokens in non_supporting_paragraphs_tokens if tokens[0] < p['idx']])
                    else:
                        num_tokens_to_add_before = \
                            sum([tokens[1] for tokens in non_supporting_paragraphs_tokens
                                 if p['idx'] > tokens[0] > supporting_paragraphs[i - 1]['idx']]) / 2

                    if i == len(supporting_paragraphs) - 1:
                        num_tokens_to_add_after = \
                            sum([tokens[1] for tokens in non_supporting_paragraphs_tokens if tokens[0] > p['idx']])
                    else:
                        num_tokens_to_add_after = \
                            sum([tokens[1] for tokens in non_supporting_paragraphs_tokens
                                 if supporting_paragraphs[i + 1]['idx'] > tokens[0] > p['idx']]) / 2

                    p['paragraph_text'] = add_wikipedia_tokens_wrap(p['paragraph_text'], p['title'],
                                                                    int(num_tokens_to_add_before),
                                                                    int(num_tokens_to_add_after),
                                                                    encoding)
                data['paragraphs'] = supporting_paragraphs
                outfile.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['original', 'noQuestion', 'expanded', 'replaced', 'oracle', 'hybrid', 'rephrase_questions'],
                        help='Name of the dataset. Must be one of: original, noQuestion, expanded, replaced, oracle, hybrid, rephrase_questions')
    parser.add_argument('--num_of_documents', type=int, default=None, help='Number of documents to include on dataset, used in hybrid')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    dataset = args.dataset_name

    if dataset == 'original':
        # this set takes the answerable questions only from the original MuSique dataset
        create_original_collection(input_path, output_path)
    elif dataset == 'noQuestion':
        # this set takes only the questions, without the documents
        create_no_questions(input_path, output_path)
    elif dataset == 'expanded':
        # this set takes only the supporting documents
        # and add the Wikipedia page content to match the original token count,
        # while keeping the original document inforamtion in the same place
        create_expanded(input_path, output_path)
    elif dataset == 'replaced':
        # this set replace the non-supporting documents with other's instances documents
        create_replace_distractors(input_path, output_path)
    elif dataset == 'oracle':
        # this set takes only the supporting documents
        create_oracle(input_path, output_path)
    elif dataset == 'hybrid':
        # this set is similar to the "expanded", but remaining more documents, as the num_of_documents provided
        create_replace_distractors(input_path, output_path, args.num_of_documents)
    elif dataset == 'rephrase_questions':
        # this set rephrased the question using GPT-4, the documents remain the same as in input
        create_rephrased_questions(input_path, output_path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset}")

    print(f"Saving {args.dataset_name} dataset in to {output_path}")

