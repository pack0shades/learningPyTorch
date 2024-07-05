import time
import random
import unidecode
import string
import re

import matplotlib.pyplot as plt
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

text_portion_size = 200

num_iter = 5000
learning_rate = 0.00001
embed_size = 100
hidden_size = 128
num_layers = 2
dropout = 0.5
batch_size = 32
with open('itstartswithusRNN.txt', 'r', encoding='utf-8') as f:
    textfile = f.read()

textfile = unidecode.unidecode(textfile)

# strip extra whitespaces
textfile = re.sub(' +',' ', textfile)

print(f'Number of characters in text: {len(textfile)}')

print(string.printable)

def portion(textfile):
    start_index = random.randint(0, len(textfile)-text_portion_size)
    end_index = start_index +text_portion_size +1
    return textfile[start_index:end_index]

print(f'================random_portion ================\n{portion(textfile)}\n===================================================')

# char to tensor
def char_to_tensor(text):
    lst = [string.printable.index(c) for c in text]
    tensor = torch.tensor(lst).long()
    return tensor

print(char_to_tensor('%$# ww'))



def draw_random_sample(textfile):
    text_long = char_to_tensor(portion(textfile))
    inputs = text_long[:-1]
    targets = text_long[1:]
    return inputs, targets

inputs, targets = draw_random_sample(textfile)
print(inputs, targets)

class RNN(torch.nn.Module):
    def __init__(self,input_size, embed_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = torch.nn.Embedding(input_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, cell):
        x = self.embed(x)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, (hidden, cell)


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(DEVICE)
        cell = torch.zeros(self.num_layers, 1, self.hidden_size).to(DEVICE)
        return (hidden, cell)


input_size = len(string.printable)

output_size = len(string.printable)

model = RNN(len(string.printable), embed_size, hidden_size, len(string.printable))
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



model = RNN(input_size, embed_size, hidden_size, output_size, num_layers, dropout)
model = model.to(DEVICE)


'''hidden, cell = model.init_hidden(batch_size)

# Example of a forward pass
inputs = torch.tensor([0]).unsqueeze(0).to(DEVICE)  # single character input
outputs, (hidden, cell) = model(inputs, hidden, cell)'''


def evaluate(model, prime_str='A', predict_len=100, temperature=0.2):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    (hidden, cell_state) = model.init_hidden(batch_size)
    prime_input = char_to_tensor(prime_str).unsqueeze(0).to(DEVICE)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        inp = prime_input[:, p].unsqueeze(1)
        _, (hidden, cell_state) = model(inp, hidden, cell_state)
    inp = prime_input[:, -1].unsqueeze(1)

    for p in range(predict_len):
        outputs, (hidden, cell_state) = model(inp, hidden, cell_state)

        # Sample from the network as a multinomial distribution
        output_dist = outputs.data.view(-1).div(temperature).exp()  # e^{logits / T}
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = string.printable[top_i]
        predicted += predicted_char
        inp = char_to_tensor(predicted_char).unsqueeze(0).to(DEVICE)

    return predicted


start_time = time.time()

loss_list = []

for iteration in range(num_iter):

    hidden, cell_state = model.init_hidden(batch_size)
    optimizer.zero_grad()

    loss = 0.
    inputs, targets = draw_random_sample(textfile)
    inputs, targets = inputs.unsqueeze(0).to(DEVICE), targets.unsqueeze(0).to(DEVICE)

    for c in range(text_portion_size):
        char_input = inputs[:, c].unsqueeze(1)
        char_targets = targets[:, c]

        outputs, (hidden, cell_state) = model(char_input, hidden, cell_state)
        loss += torch.nn.functional.cross_entropy(outputs, char_targets)

    loss /= text_portion_size
    loss.backward()

    ### UPDATE MODEL PARAMETERS
    optimizer.step()

    ### LOGGING
    with torch.no_grad():
        if iteration % 200 == 0:
            print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
            print(f'Iteration {iteration} | Loss {loss.item():.2f}\n\n')
            print(evaluate(model, 'Th', 200), '\n')
            print(50 * '=')

            loss_list.append(loss.item())
            plt.clf()
            plt.plot(range(len(loss_list)), loss_list)
            plt.ylabel('Loss')
            plt.xlabel('Iteration x 1000')
            plt.savefig('loss1.pdf')

plt.clf()
plt.ylabel('Loss')
plt.xlabel('Iteration x 1000')
plt.plot(range(len(loss_list)), loss_list)
plt.show()
