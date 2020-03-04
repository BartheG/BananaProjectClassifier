from Prediction.m_predict import m_predict
import json

''' Example of m_predict usage '''

def main():
	f_pred = m_predict()
	print(f_pred( './DataTest/desbananes.jpg',False )[0])
	print(f_pred( './DataTest/bananepasquali.jpg',False )[0])
	print(f_pred( './DataTest/bananequali.jpeg',False )[0])

if __name__ == "__main__":
	main()