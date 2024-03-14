# searchGUI.py - GUI for search
# AIFCA Python code Version 0.9.12 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

class SearcherGUI(object):
    def __init__(self, SearchClass, problem, fontsize=10,
                     colors = {'selected':'red', 'neighbors':'blue', 'frontier':'green', 'goal':'yellow'}):
        self.problem = problem
        self.searcher = SearchClass(problem)
        self.problem.fontsize = fontsize
        self.colors = colors
        #self.go()

    def go(self):
        fig, self.ax = plt.subplots()
        plt.ion()   # interactive
        self.ax.set_axis_off()
        plt.subplots_adjust(bottom=0.15)
        step_butt = Button(plt.axes([0.05,0.02,0.15,0.05]), "step")
        step_butt.on_clicked(self.step)
        fine_butt = Button(plt.axes([0.25,0.02,0.15,0.05]), "fine step")
        fine_butt.on_clicked(self.finestep)
        auto_butt = Button(plt.axes([0.45,0.02,0.15,0.05]), "auto search")
        auto_butt.on_clicked(self.auto)
        quit_butt = Button(plt.axes([0.65,0.02,0.15,0.05]), "quit")
        quit_butt.on_clicked(self.quit)
        self.ax.text(0.85,0, '\n'.join(self.colors[a]+": "+a for a in self.colors))
        self.problem.show_graph(self.ax, node_color='white')
        self.problem.show_node(self.ax, self.problem.start, self.colors['frontier'])
        for node in self.problem.nodes:
            if self.problem.is_goal(node):
                self.problem.show_node(self.ax, node,self.colors['goal'])
        plt.show()
        self.click = 7 # bigger than any display!
        #while self.click == 0:
        #    plt.pause(0.1)
        self.searcher.display = self.display
        try:
            while self.searcher.frontier:
                path = self.searcher.search()
        except ExitToPython:
            print("Exited")
        else:
            print("No more solutions")
                        
    def display(self, level,*args,**nargs):
        if level <= self.click: #step
            print(*args, **nargs)
            self.ax.set_title(f"Expanding: {self.searcher.path}",fontsize=self.problem.fontsize)
            if level == 1:
                self.show_frontier(self.colors['frontier'])
                self.show_path(self.colors['selected'])
                self.ax.set_title(f"Solution Found: {self.searcher.path}",fontsize=self.problem.fontsize)
            elif level == 2: # what should be shown if a node is in all three?
                self.show_frontier(self.colors['frontier'])
                self.show_path(self.colors['selected'])
                self.show_neighbors(self.colors['neighbors'])
            elif level == 3:
                self.show_frontier(self.colors['frontier'])
                self.show_path(self.colors['selected'])
            elif level == 4:
                self.show_frontier(self.colors['frontier'])


            # wait for a button click
            self.click = 0
            plt.draw()
            while self.click == 0:
                plt.pause(0.1)
            # undo coloring:
            self.ax.set_title("")
            self.show_frontier('white')
            self.show_neighbors('white')
            path_show = self.searcher.path
            while path_show.arc:
                self.problem.show_arc(self.ax, path_show.arc, 'black')
                self.problem.show_node(self.ax, path_show.end(), 'white')
                path_show = path_show.initial
            self.problem.show_node(self.ax, path_show.end(), 'white')
            if self.problem.is_goal(self.searcher.path.end()):
                self.problem.show_node(self.ax, self.searcher.path.end(), self.colors['goal'])
            plt.draw()

    def show_frontier(self, color):
        for path in self.searcher.frontier:
            self.problem.show_node(self.ax, path.end(), color)
                
    def show_path(self, color):
        """color selected path"""
        path_show = self.searcher.path
        while path_show.arc:
                self.problem.show_arc(self.ax, path_show.arc, color)
                self.problem.show_node(self.ax, path_show.end(), color)
                path_show = path_show.initial
        self.problem.show_node(self.ax, path_show.end(), color)

    def show_neighbors(self, color):
        for neigh in self.problem.neighbors(self.searcher.path.end()):
            self.problem.show_node(self.ax, neigh.to_node, color)

    def auto(self,event):
        self.click = 1
    def step(self,event):
        self.click = 2
    def finestep(self,event):
        self.click = 3
    def quit(self,event):
        quit()

class ExitToPython(Exception):
    pass

from searchGeneric import Searcher, AStarSearcher
from searchMPP import SearcherMPP
import searchExample
from searchBranchAndBound import DF_branch_and_bound

# to demonstrate depth-first search:
# sdfs = SearcherGUI(Searcher, searchExample.tree_graph); sdfs.go()

# delivery graph examples:
# sh = SearcherGUI(Searcher, searchExample.simp_delivery_graph); sh.go()
# sha = SearcherGUI(AStarSearcher, searchExample.simp_delivery_graph); sha.go()
# shac = SearcherGUI(AStarSearcher, searchExample.cyclic_simp_delivery_graph); shac.go()
# shm = SearcherGUI(SearcherMPP, searchExample.cyclic_simp_delivery_graph); shm.go()
# shb = SearcherGUI(DF_branch_and_bound, searchExample.simp_delivery_graph); shb.go()

# The following is AI:FCA figure 3.15, and is useful to show branch&bound:
# shbt = SearcherGUI(DF_branch_and_bound, searchExample.tree_graph); shbt.go()

