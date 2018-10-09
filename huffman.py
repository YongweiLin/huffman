"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    result = {}
    for item in text:
        if item not in result:
            result[item] = 1
        else:
            result[item] += 1
    return result


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    # For this function, we read
    # https://www.huffmancoding.com/my-uncle/huffman-algorithm
    node_list = []
    # Create a list of tuples of leaf node and corresponding frequency.
    for key in freq_dict:
        node_list.append((HuffmanNode(key), freq_dict[key]))
    # Special case of just one symbol in dict.
    if len(node_list) == 1:
        only_node = node_list[0][0]
        dummy_symbol = only_node.symbol + 1
        if dummy_symbol == 256:
            dummy_symbol = 0
        dummy_node = HuffmanNode(dummy_symbol)
        parent_node = HuffmanNode(None, only_node, dummy_node)
        return parent_node
    # Case of mutiple symbols.
    while len(node_list) > 1:
        first_choice = get_lowest_freq_node(node_list)
        second_choice = get_lowest_freq_node(node_list)
        parent_node = HuffmanNode(None, first_choice[0], second_choice[0])
        node_list.append((parent_node, first_choice[1] + second_choice[1]))
    return node_list[0][0]


def get_lowest_freq_node(node_list):
    """ Return a tuple containing node with lowest frequency from node_list
    and pop out this tuple from node_list.

    @param list node_list: list of tuples with node and frequency
    @rtype: tuple

    >>> node_list = [(HuffmanNode(3), 4), (HuffmanNode(2), 6)]
    >>> get_lowest_freq_node(node_list)
    (HuffmanNode(3, None, None), 4)
    >>> node_list == [(HuffmanNode(2), 6)]
    True
    """
    result = 0
    i = 1
    while i < len(node_list):
        if node_list[i][1] < node_list[result][1]:
            result = i
        i += 1
    temp = node_list[result]
    node_list.remove(node_list[result])
    return temp


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    result = {}
    assign_code(tree, result, '')
    return result


def assign_code(tree, code_dict, current_code):
    """ Mutate code_dict by adding symbols with their new codes and mutate
    current_code by appending 0 or 1, according to tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict code_dict: symbol-to-code dictionary
    @param str current_code: a string only containing 0 or 1
    @rtype: None

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> code_dict = {}
    >>> assign_code(tree, code_dict, '')
    >>> code_dict == {3: "0", 2: "1"}
    True
    """
    if tree.is_leaf():
        code_dict[tree.symbol] = current_code
    else:
        assign_code(tree.left, code_dict, current_code + '0')
        assign_code(tree.right, code_dict, current_code + '1')


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    num = [0]
    assign_number(tree, num)


def assign_number(tree, num):
    """ Mutate tree by assigning the integer in num to the internal nodes
    in postorder traversal, and the integer in num increase by 1 when arrives
    next internal node.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param list num: a list with one integer which keeps track of the order
    of postorder traversal.
    @rtype: None

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> assign_number(tree, [0])
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    if tree.is_leaf():
        pass
    else:
        assign_number(tree.left, num)
        assign_number(tree.right, num)
        tree.number = num[0]
        num[0] += 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    total_len = 0
    total_freq = 0
    code_d = get_codes(tree)
    if len(code_d) == 2 and len(freq_dict) == 1:
        for key in freq_dict:
            total_len += len(code_d[key]) * freq_dict[key]
            total_freq += freq_dict[key]
        return total_len / total_freq
    else:
        for key in code_d:
            total_len += len(code_d[key]) * freq_dict[key]
            total_freq += freq_dict[key]
        return total_len / total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bit_list = []
    bit_string = ''
    # Combine all the code together into a long string.
    for byte in text:
        bit_string += codes[byte]
    # Case of less than or equal to 8 bits, just append.
    if len(bit_string) <= 8:
        bit_list.append(bit_string)
    # Case of longer string.
    else:
        start_index = 0
        end_index = 8
        # Append a 8-bit string in every iteration.
        while end_index < len(bit_string):
            bit_list.append(bit_string[start_index: end_index])
            start_index = end_index
            end_index += 8
        # Append the string left in the end.
        bit_list.append(bit_string[start_index:])
    # Fill the last string in the list if it is not 8-bit long.
    if len(bit_list[-1]) < 8:
        for _ in range(8 - len(bit_list[-1])):
            bit_list[-1] += '0'
    # Translate all 8-bit strings into bytes.
    for i in range(len(bit_list)):
        bit_list[i] = bits_to_byte(bit_list[i])
    result = bytes(bit_list)
    return result


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    if tree.is_leaf():
        return bytes([])
    else:
        result = bytes([])
        result += tree_to_bytes(tree.left)
        result += tree_to_bytes(tree.right)
        acc = []
        if isinstance(tree.left.symbol, int):
            acc.append(0)
            acc.append(tree.left.symbol)
        else:
            acc.append(1)
            acc.append(tree.left.number)
        if isinstance(tree.right.symbol, int):
            acc.append(0)
            acc.append(tree.right.symbol)
        else:
            acc.append(1)
            acc.append(tree.right.number)
        return result + bytes(acc)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root = HuffmanNode()
    read_root = node_lst[root_index]
    # consider left of the root node.
    if read_root.l_type == 0:
        root.left = HuffmanNode(read_root.l_data)
    else:
        root.left = generate_tree_general(node_lst, read_root.l_data)
    # consider right of the root node.
    if read_root.r_type == 0:
        root.right = HuffmanNode(read_root.r_data)
    else:
        root.right = generate_tree_general(node_lst, read_root.r_data)
    return root


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    root_node_read = node_lst[root_index]
    root = HuffmanNode()
    # if the root node has two leaves.
    if (root_node_read.l_type, root_node_read.r_type) == (0, 0):
        root.left = HuffmanNode(root_node_read.l_data)
        root.right = HuffmanNode(root_node_read.r_data)
        return root
    else:
        # if root node has left leaf and right subtree.
        if (root_node_read.l_type, root_node_read.r_type) == (0, 1):
            root.left = HuffmanNode(root_node_read.l_data)
            root.right = generate_tree_postorder(node_lst, root_index - 1)
        # if root node has right leaf and left subtree.
        if (root_node_read.l_type, root_node_read.r_type) == (1, 0):
            root.left = generate_tree_postorder(node_lst, root_index - 1)
            root.right = HuffmanNode(root_node_read.r_data)
        # if both left and right have subtrees.
        if (root_node_read.l_type, root_node_read.r_type) == (1, 1):
            # construct the right subtree.
            root.right = generate_tree_postorder(node_lst, root_index - 1)
            # find the root index of the left subtree.
            left_root_index = root_index - count_internal(root.right) - 1
            root.left = generate_tree_postorder(node_lst, left_root_index)
        return root


def count_internal(node):
    """ Return the number of internal node in the Huffman tree corresponding
    to the HuffmanNode Node.

    @param HuffmanNode node: the Huffman node to generate Huffman tree.
    @rtype: int

    >>> t = HuffmanNode(3, None, None)
    >>> count_internal(t)
    0
    >>> t = HuffmanNode(3, HuffmanNode(2, None, None), None)
    >>> count_internal(t)
    1
    """
    if node is None:
        return 0
    elif node.is_leaf():
        return 0
    else:
        acc = 1
        acc += count_internal(node.left)
        acc += count_internal(node.right)
        return acc


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    d = uncompressed_helper(tree)
    bit_str = ''
    for byte in text:
        bit_text = byte_to_bits(byte)
        bit_str += bit_text
    result = []
    i = 0
    j = 1
    # when there is still bit left
    while len(result) < size:
        # if find a code
        if bit_str[i:j] in d:
            # add the correspond original text to result
            result.append(d[bit_str[i:j]])
            # move the start slicing index to the last time end point
            i = j
            # move the end slicing index to one after
            j += 1
        # if not find a code
        else:
            # move the end slicing index to one after
            j += 1
    return bytes(result)


def uncompressed_helper(tree):
    """ Return a dictionary whose keys are codes and values are corresponding
    symbols in tree.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @rtype: dict(str,int)

    >>> tree = HuffmanNode(None, HuffmanNode(1), HuffmanNode(2))
    >>> uncompressed_helper(tree) == {'1': 2, '0': 1}
    True
    """
    symbol_to_code = get_codes(tree)
    code_to_symbol = {}
    for key in symbol_to_code:
        code_to_symbol[symbol_to_code[key]] = key
    return code_to_symbol


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # For this function, we watched https://www.youtube.com/watch?v=86g8jAQug04.
    # We find queue and level order traversal can be helpful for this function.
    if tree is not None:
        # reverse the place of frequency and symbol for the original dictionary.
        reverse_d = {}
        for key in freq_dict:
            if freq_dict[key] not in reverse_d:
                reverse_d[freq_dict[key]] = [key]
            else:
                reverse_d[freq_dict[key]].append(key)
        # creat a list of symbols from the least frequency to the highest
        # frequency
        freq_list = []
        for freq in reverse_d:
            for _ in range(len(reverse_d[freq])):
                freq_list.append(freq)
        freq_list = sorted(freq_list)
        # use level order traversal to examine each leaf and change them
        # according to the frequency of occurence of symbols based on
        # the sorted freq_list
        queue = [tree]
        while queue:
            temp = queue.pop(0)
            if temp.symbol is None:
                queue.append(temp.left)
                queue.append(temp.right)
            else:
                temp.symbol = reverse_d[freq_list[-1]][-1]
                reverse_d[freq_list[-1]].pop()
                freq_list.pop()


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
