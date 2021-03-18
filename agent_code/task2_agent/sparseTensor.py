# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:17:09 2021

@author: Philipp
"""
import numpy as np


class SparseTensor:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.values = []
        self.indices = []

    def get_all(self):
        return self.values

    def add_entry(self, index, value):
        if self.already_exists(index):
            print("WARNING: add_entry got a index that already exists in tuple")
        if len(self.dimensions) != len(index):
            raise Exception(
                "ERROR: input size does not match dimension of sparse tensor"
            )
            return

        for i in range(len(self.dimensions)):
            if self.dimensions[i] < index[i] or index[i] < 0:
                raise Exception(
                    "ERROR: OutOfBounds in SparseTuple when trying to add -> ", index
                )

        self.values.append(value)
        self.indices.append(index)

    def get_last_splice(self, index):
        assert len(index) - 1 != len(self.dimensions)

        splice = []
        for i in range(self.dimensions[-1]):
            if self.already_exists(np.concatenate((index, [i]))):
                splice.append(self.get_entry(np.concatenate((index, [i]))))
            else:
                splice.append(0)
        return splice

    def get_entry(self, index):
        for i in range(len(self.dimensions)):
            if self.dimensions[i] < index[i] or index[i] < 0:
                raise Exception(
                    "ERROR: OutOfBounds in SparseTuple when trying to get -> ", index
                )
        return self.values[
            [np.array_equal(index, ind) for ind in self.indices].index(True)
        ]

    def already_exists(self, index):
        for i in range(len(self.dimensions)):
            if self.dimensions[i] < index[i] or index[i] < 0:
                raise Exception(
                    "ERROR: OutOfBounds in SparseTuple when trying to check if already exists -> ",
                    index,
                )
        return any(np.array_equal(index, ind) for ind in self.indices)

    def change_value(self, index, new_value):
        for i in range(len(self.dimensions)):
            if self.dimensions[i] < index[i] or index[i] < 0:
                raise Exception(
                    "ERROR: OutOfBounds in SparseTuple when trying to change value at -> ",
                    index,
                )
        if any(np.array_equal(index, ind) for ind in self.indices):
            self.indices[
                [np.array_equal(index, ind) for ind in self.indices].index(True)
            ] = new_value
        else:
            raise Exception(
                "ERROR: change_value gets a index that was not added to the tuple beforehand"
            )
