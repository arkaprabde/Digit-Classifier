import pandas as pd
from matplotlib import pyplot as plt
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import cupy as cp

c = 0
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.button_clear = tk.Button(self.root, text='Clear', command=self.clear_canvas)
        self.button_clear.pack()
        
        self.button_predict = tk.Button(self.root, text='Predict', command=self.predict_digit)
        self.button_predict.pack()
        
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.image = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        global c
        digit_image = self.image.resize((28, 28)).convert('L')
        digit_image = ImageOps.invert(digit_image)
        digit_data = cp.array(digit_image) / 255.0
        digit_data = digit_data.reshape((784, 1))
        
        img = cp.asnumpy(digit_data.reshape((28, 28)) * 255)
        plt.gray()
        plt.imshow(img, interpolation = 'nearest')
        c += 1
        plt.savefig(os.path.join('pics', f'pic{c}.png'))
        
        W1, b1, W2, b2, W3, b3 , W4, b4 = load_params()
        A = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, digit_data)
        prediction = cp.argmax(A, axis=0)
        print(f"Predicted Digit: {prediction[0]}")

def ReLU(Z):
    return cp.maximum(0, Z)

def Tanh(Z):
    return cp.tanh(Z)

def Softmax(Z):
    expZ = cp.exp(Z - cp.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = Tanh(Z2)
    
    Z3 = W3.dot(A2) + b3
    A3 = ReLU(Z3)
    
    Z4 = W4.dot(A3) + b4
    A4 = Softmax(Z4)
    
    return A4

def load_params(folder_path='params'):
    W1 = cp.loadtxt(os.path.join(folder_path, 'W1.txt'))
    b1 = cp.loadtxt(os.path.join(folder_path, 'b1.txt'))
    W2 = cp.loadtxt(os.path.join(folder_path, 'W2.txt'))
    b2 = cp.loadtxt(os.path.join(folder_path, 'b2.txt'))
    W3 = cp.loadtxt(os.path.join(folder_path, 'W3.txt'))
    b3 = cp.loadtxt(os.path.join(folder_path, 'b3.txt'))
    W4 = cp.loadtxt(os.path.join(folder_path, 'W4.txt'))
    b4 = cp.loadtxt(os.path.join(folder_path, 'b4.txt'))
    
    b1 = b1.reshape(W1.shape[0], 1)
    b2 = b2.reshape(W2.shape[0], 1)
    b3 = b3.reshape(W3.shape[0], 1)
    b4 = b4.reshape(W4.shape[0], 1)
    return W1, b1, W2, b2, W3, b3, W4, b4

def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
