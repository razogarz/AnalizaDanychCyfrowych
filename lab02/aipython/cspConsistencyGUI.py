# cspConsistencyGUI.py - GUI for consistency-based CSP solving
# AIFCA Python code Version 0.9.12 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from cspConsistency import Con_solver
import matplotlib.pyplot as plt

class ConsistencyGUI(Con_solver):
    def __init__(self, csp, fontsize=10, speed=1, **kwargs):
        """
        csp is the csp to show
        fontsize is the size of the text
        speed is the number of animations per second (controls delay_time)
             1 (slow) and 4 (fast) seem like good values
        """
        self.fontsize = fontsize
        self.delay_time = 1/speed
        Con_solver.__init__(self, csp, **kwargs)
        csp.show(showAutoAC = True)

    def go(self):
        res = self.solve_all()
        self.csp.draw_graph(domains=self.domains, 
                                title="No more solutions. GUI finished. ",
                                fontsize=self.fontsize)
        return res

    def select_arc(self, to_do):
        while True:
            self.csp.draw_graph(domains=self.domains, to_do=to_do,
                                    title="click on to_do (blue) arc", fontsize=self.fontsize)
            while self.csp.picked == None and not self.csp.autoAC:
                plt.pause(0.01) # controls reaction time of GUI
            if self.csp.autoAC:
                break
            picked = self.csp.picked
            self.csp.picked = None
            if picked in to_do:
                to_do.remove(picked)
                print(f"{picked} picked")
                return picked
            else:
                print(f"{picked} not in to_do")
        if self.csp.autoAC:
            self.csp.draw_graph(domains=self.domains, to_do=to_do,
                                    title="Auto AC", fontsize=self.fontsize)
            plt.pause(self.delay_time)
            return to_do.pop() 

    def select_var(self, iter_vars):
        vars = list(iter_vars)
        while True:
            self.csp.draw_graph(domains=self.domains, 
                                    title="Arc consistent. Click node to split",
                                    fontsize=self.fontsize)
            while self.csp.picked == None:
                plt.pause(0.01) # controls reaction time of GUI
            picked = self.csp.picked
            self.csp.picked = None
            self.csp.autoAC = False
            if picked in vars:
                #print("splitting",picked)
                return picked
            else:
                print(picked,"not in",vars)

    def display(self,n,*args,**nargs):
        if n <= self.max_display_level:  # default display
            print(*args, **nargs)
        if n==1: # solution found or no solutions"
            self.csp.draw_graph(domains=self.domains, to_do=set(),
                                    title=' '.join(args)+": click any node or arc to continue",
                                    fontsize=self.fontsize)
            self.csp.autoAC = False
            while self.csp.picked == None and not self.csp.autoAC:
                plt.pause(0.01) # controls reaction time of GUI
            self.csp.picked = None
        elif n==2: # backtracking
            plt.title("backtracking: click any node or arc to continue")
            self.csp.autoAC = False
            while self.csp.picked == None  and not self.csp.autoAC:
                plt.pause(0.01)
            self.csp.picked = None
        elif n==3:  # inconsistent arc
            line = self.csp.thelines[self.arc_selected]
            line.set_color('red')
            line.set_linewidth(10)
            plt.pause(self.delay_time)
            line.set_color('limegreen')
            line.set_linewidth(self.csp.linewidth)
        #elif n==4 and self.add_to_do: # adding to to_do
        #    print("adding to to_do",self.add_to_do)  ## highlight these arc

import cspExamples
# Try:
# ConsistencyGUI(cspExamples.csp1).go()
# ConsistencyGUI(cspExamples.csp3).go()
# ConsistencyGUI(cspExamples.csp3, speed=4, fontsize=15).go()

