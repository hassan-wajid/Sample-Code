// Code Sample 2

#include<iostream>
using namespace std;

// BST = BinarySearchTree
class BSTNode
{
public:
	int key;
	BSTNode* left, * right;
	BSTNode(int el = 0, BSTNode* l = NULL, BSTNode * r = NULL)
	{
		key = el;
		left = l;
		right = r;
	}
};

class BST
{
public:
	BSTNode* root;

	BST()
	{
		root = NULL;
	}

	//destroy routines
	~BST();

	void clear(BSTNode* ptr);

	//insertion routines
	void insert(int A);
	void insert_node(int B, BSTNode*& node1);

	//traversal routines
	void visit(BSTNode* node)
	{
		cout << "::" << node->key << "::" << endl;
	}

	void breadth_first();
	void depth_first();

	void preorder(BSTNode* node);
	void inorder(BSTNode* node);

	//searching routines
	BSTNode* search(int A);
	BSTNode* search_node(BSTNode* node, int B);

	//deletion routines
	void deletenode(int A);
	void deletebycopying(BSTNode*& node2);
};

void BST::inorder(BSTNode* node)
{
	if (node != NULL)
	{
		inorder(node->left);
		visit(node);
		inorder(node->right);
	}
}

void BST::preorder(BSTNode* node)
{
	if (node != NULL)
	{
		visit(node);
		preorder(node->left);
		preorder(node->right);
	}
}

void BST::depth_first()
{
	//preorder(root);
	inorder(root);
}

void BST::deletenode(int A)
{

	BSTNode* node = root, * prev = NULL;

	while (node != NULL)
	{
		if (node->key == A)
			break;

		prev = node;

		if (A < node->key)
			node = node->left;

		else
			node = node->right;
	}

	if (node != NULL && node->key == A)
	{
		if (node == root)
			deletebycopying(root);

		else if (prev->left == node)
			deletebycopying(prev->left);

		else
			deletebycopying(prev->right);
	}

	else
		cout << "Node:" << A << "is not found\n";
}

void BST::deletebycopying(BSTNode*& node2)
{
	BSTNode* prev, * temp;

	prev = temp = node2;


	if (node2->right == NULL)		//node has no right child, or it is a leaf
	{
		node2 = node2->left;
		delete temp;
	}
	else if (node2->left == NULL)	//node has no left child, or it is a leaf
	{
		node2 = node2->right;
		delete temp;
	}
	else							//node has both left and right child
	{
		prev = node2;

		//left subtree
		temp = node2->left;

		//go to the right most node in the left subtree
		while (temp->right != NULL)
		{
			prev = temp;
			temp = temp->right;
		}

		//copy the value in the node to be deleted
		node2->key = temp->key;

		if (prev == node2)
			prev->left = temp->left;

		else
			prev->right = temp->left;

		delete temp;
	}
}

BSTNode* BST::search_node(BSTNode* node, int B)
{
	if (node != NULL)
	{
		if (node->key == B)
			return node;

		else if (B < node->key)
			search_node(node->left, B);

		else if (B >= node->key)
			search_node(node->right, B);
	}
}

BSTNode* BST::search(int A)
{
	BSTNode* temp = NULL;
	temp = search_node(root, A);

	return temp;
}

void BST::insert_node(int B, BSTNode* & node1)
{
	if (node1 == NULL)
	{
		node1 = new BSTNode(B);
	}
	else if(B < node1->key)
		insert_node(B, node1->left);
	else if(B >= node1->key)
		insert_node(B, node1->right);
}

void BST::insert(int A)
{
	insert_node(A, root);
}

void BST::clear(BSTNode* ptr)
{
	if (ptr != NULL)
	{
		clear(ptr->left);
		clear(ptr->right);

		delete ptr;
	}
}

BST::~BST()
{
	clear(root);
}
BSTNode* searchTreeNode(BSTNode* root, int data) {
    if(root == NULL){
        cout<<"node not found";
    }else{
        if(root->key == data){
            cout<<"node found";
        }else if(root->key >= data){
            root = searchTreeNode(root->left,data);
        }else if(root->key < data){
            root = searchTreeNode(root->right, data);
        }
    }
    return root;
}
BSTNode* findMinFromRight(BSTNode* node) {
    while(node->left != NULL){
        node = node->left;
    }
    return node;
}

 BSTNode* Successor(BSTNode* root, int data) {
    cout<<"called ";
    BSTNode *keyNode = searchTreeNode(root, data);
    if(keyNode == NULL){
        return keyNode;
    }else{
        if(keyNode->right != NULL){
            root = findMinFromRight(keyNode->right);
            return root;
        }else{
            BSTNode* suc = NULL;
            BSTNode* anc = root;
            while(anc != keyNode){
                if(anc->key > keyNode->key){
                    suc = anc;
                    anc = anc->left;
            }else{
                    anc = anc->right;
                }
            }
            return suc;
        }
    }
}

int main()
{
	BST Tree1;

	Tree1.insert(5);
	Tree1.insert(2);
	Tree1.insert(11);
	Tree1.insert(1);
	Tree1.insert(4);
	Tree1.insert(3);

	Tree1.insert(6);
	Tree1.insert(18);
	Tree1.insert(14);
	Tree1.insert(16);
	Tree1.insert(15);
	Tree1.insert(25);
	Tree1.insert(22);
	Tree1.insert(21);
	Tree1.insert(23);
	Tree1.insert(28);
	BSTNode *t;
	t=Successor(Tree1.root,28);
    cout<<t->key;
}
