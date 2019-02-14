1.	Url
2.	tokenized clean body. This column is a json array in which each element is a token.
3.	keyphrase start/end index (inclusive). It’s a Json array, each element is a pair of keyphrase position index.
4.	Vdom features for each text node. Detailed format is as below.

Each text node’s vdom feature contains 
1.	the start/end index of this node in cleanbody
2.	node ID (dummy value for now)
3.	vdom feature vector:
a.	10 features for this text node and its parent node: (20 values in total) 
i.	Node’ x, y coordinates on this page
ii.	Node’ height and width
iii.	Is this node a block element
iv.	Is this node an inline element
v.	Is this node a heading tag element
vi.	Is this node a leaf element
vii.	Node font size
viii.	Is this node bolded

