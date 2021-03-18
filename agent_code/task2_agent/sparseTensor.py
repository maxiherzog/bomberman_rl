# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:17:09 2021

@author: Philipp
"""
class SparseTensor:
	
	def __init__(self, dimension):
		self.values = ()
		self.indices = ()
		self.current_index = 0
		self.dimension = dimension.size
		self.len = dimension
	
	def add_entry(self, value, index):
		if self.already_exists(index):
			print('WARNING: add_entry got a index that already exists in tuple')
		if self.dimension != index.size:
			raise Exception('ERROR: input size does not match dimension of sparse tensor')
			return
		
		for i in range(len(self.len)):
			if(self.len[i] < index[i] or index[i] < 0):
				raise Exception('ERROR: OutOfBounds in SparseTuple when trying to add -> ', index)
				
		self.values[self.current_index] = value
		self.indices[self.current_index] = index
		self.current_index += 1
		
	def get_entry(self, index):
		for i in range(len(self.len)):
			if(self.len[i] < index[i] or index[i] < 0):
				raise Exception('ERROR: OutOfBounds in SparseTuple when trying to get -> ', index)
		return self.values[self.indices.index(index)]#, self.indices[self.indices.index(index)]
	
	def already_exists(self, index):
		for i in range(len(self.len)):
			if(self.len[i] < index[i] or index[i] < 0):
				raise Exception('ERROR: OutOfBounds in SparseTuple when trying to check if already exists -> ', index)
		return index in self.indices
		
	def change_value(self, index, new_value):
		for i in range(len(self.len)):
			if(self.len[i] < index[i] or index[i] < 0):
				raise Exception('ERROR: OutOfBounds in SparseTuple when trying to change value at -> ', index)
		if(index in self.indices):
			self.indices[self.indices.index(index)] = new_value
		else:
			raise Exception('ERROR: change_value gets a index that was not added to the tuple beforehand')