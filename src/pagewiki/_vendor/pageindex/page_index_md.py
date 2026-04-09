"""Markdown → hierarchical tree builder, vendored from VectifyAI/PageIndex.

Upstream: pageindex/page_index_md.py @ f2dcffc
License:  MIT (see ./LICENSE)

## Modifications from upstream

Only the pure-Python tree-building helpers are kept. The removed
symbols were all LLM- or I/O-dependent and are replaced in pagewiki by
the ``pageindex_adapter`` module (which uses the injected ``chat_fn``):

  * ``get_node_summary``
  * ``generate_summaries_for_structure_md``
  * ``md_to_tree``
  * the ``__main__`` smoke test block

The remaining functions are verbatim from upstream aside from
``from .utils import *`` being narrowed to ``from .utils import count_tokens``.
"""

from __future__ import annotations

import re

from .utils import count_tokens


def extract_nodes_from_markdown(markdown_content):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []

    lines = markdown_content.split('\n')
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()

        # Check for code block delimiters (triple backticks)
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue

        # Skip empty lines
        if not stripped_line:
            continue

        # Only look for headers when not inside a code block
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({'node_title': title, 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list, markdown_lines):
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)

        if header_match is None:
            continue

        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1)),
        }
        all_nodes.append(processed_node)

    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1
        else:
            end_line = len(markdown_lines)

        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()
    return all_nodes


def update_node_list_with_text_token_count(node_list, model=None):
    def find_all_children(parent_index, parent_level, node_list):
        """Find all direct and indirect children of a parent node."""
        children_indices = []
        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]['level']
            if current_level <= parent_level:
                break
            children_indices.append(i)
        return children_indices

    result_list = node_list.copy()

    # Process nodes from end to beginning so children are counted before parents
    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        current_level = current_node['level']

        children_indices = find_all_children(i, current_level, result_list)

        node_text = current_node.get('text', '')
        total_text = node_text
        for child_index in children_indices:
            child_text = result_list[child_index].get('text', '')
            if child_text:
                total_text += '\n' + child_text

        result_list[i]['text_token_count'] = count_tokens(total_text, model=model)

    return result_list


def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    def find_all_children(parent_index, parent_level, node_list):
        children_indices = []
        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]['level']
            if current_level <= parent_level:
                break
            children_indices.append(i)
        return children_indices

    result_list = node_list.copy()
    nodes_to_remove = set()

    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue

        current_node = result_list[i]
        current_level = current_node['level']

        total_tokens = current_node.get('text_token_count', 0)

        if min_node_token is not None and total_tokens < min_node_token:
            children_indices = find_all_children(i, current_level, result_list)

            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get('text', '')
                    if child_text.strip():
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)

            if children_texts:
                parent_text = current_node.get('text', '')
                merged_text = parent_text
                for child_text in children_texts:
                    if merged_text and not merged_text.endswith('\n'):
                        merged_text += '\n\n'
                    merged_text += child_text

                result_list[i]['text'] = merged_text
                result_list[i]['text_token_count'] = count_tokens(merged_text, model=model)

    for index in sorted(nodes_to_remove, reverse=True):
        result_list.pop(index)

    return result_list


def build_tree_from_nodes(node_list):
    if not node_list:
        return []

    stack = []
    root_nodes = []
    node_counter = 1

    for node in node_list:
        current_level = node['level']

        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': [],
        }
        node_counter += 1

        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node['nodes'].append(tree_node)

        stack.append((tree_node, current_level))

    return root_nodes


def clean_tree_for_output(tree_nodes):
    cleaned_nodes = []

    for node in tree_nodes:
        cleaned_node = {
            'title': node['title'],
            'node_id': node['node_id'],
            'text': node['text'],
            'line_num': node['line_num'],
        }

        if node['nodes']:
            cleaned_node['nodes'] = clean_tree_for_output(node['nodes'])

        cleaned_nodes.append(cleaned_node)

    return cleaned_nodes
