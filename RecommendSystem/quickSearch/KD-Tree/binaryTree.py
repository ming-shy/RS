# 构建二叉排序树

# 首先定义一个树结构
class TreeStruct:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right


# 创建KD-Tree
class BTree:
    def __init__(self, root):
        self.root = root

    def insert(self, data):
        self.insertNode(data, self.root)

    def insertNode(self, data, rootNode:TreeStruct):
        if rootNode is None:
            self.root = TreeStruct(data, None, None)
        elif data < rootNode.val:
            if rootNode.left is None:
                rootNode.left = TreeStruct(data, None, None)
                return
            else:
               self.insertNode(data, rootNode.left)
        elif data > rootNode.val:
            if rootNode.right is None:
                rootNode.right = TreeStruct(data, None, None)
                return
            else:
                self.insertNode(data, rootNode.right)

    def printTree(self):
        self.printTreeImpl(self.root)

    def printTreeImpl(self, node):
        if node is None:
            # print("该树为空......")
            return
        else:
            self.printTreeImpl(node.left)
            print(node.val)
            self.printTreeImpl(node.right)


if __name__ == '__main__':
    node_val = [3, 2, 4, 1, 9, 6, 18]
    kdTree = BTree(None)
    for node in node_val:
        kdTree.insert(node)
    kdTree.printTree()
