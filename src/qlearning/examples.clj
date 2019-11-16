(ns qlearning.examples
  (:require [qlearning.rl :refer [learnQ optimal-policy generate-grid-world show-Q-table]]))

;;;******************************************************************
;;; Q-Learning Grid World Examples
;;;******************************************************************

;;;------------------------------------------------------------------
;;; Example 1: Simple Grid World example from Tom Mitchell's ML book.
;;;------------------------------------------------------------------

;;; Initial state is :s1 and goal stat is :s3.

(def simple-grid
  {:s1 {:east [:s2 0], :south [:s4 0]}
   :s2 {:east [:s3 100], :west [:s1 0], :south [:s5 0]}
   :s3 {:east [:s3 0], :west [:s3 0], :north [:s3 0] :south [:s3 0]}
   :s4 {:east [:s5 0], :north [:s1 0]}
   :s5 {:east [:s6 0], :west [:s4 0], :north [:s2 0]}
   :s6 {:east [:s5 0], :north [:s3 100]}})

;;;------------------------------------------------------------------

(defn ex1 []
  (let [qt (learnQ :s1 :s3 simple-grid 100)
        op (optimal-policy :s1 :s3 qt)]
    (print "\nLearning Q values for:\n")
    (show-Q-table simple-grid)
    (print "\n\nResulting Q values after " 100 " iterations:\n")
    (show-Q-table @qt)
    (print "\n\nOptimal policy is:\n")
    (print op)))

;;;------------------------------------------------------------------
;;; Example 2: 4x4 Grid World
;;;------------------------------------------------------------------

;;; Domain: 4 x 4 Grid, Start: top-left, Goal: bottom-right.

(defn ex2 []
  (let [grid-world (generate-grid-world 4 4 :s-4-4)
        qt (learnQ :s-1-1 :s-4-4 grid-world 2000)
        op (optimal-policy :s-1-1 :s-4-4 qt)]
    (print "\nLearning Q values for:\n")
    (show-Q-table grid-world)
    (print "\n\nResulting Q values after 2000 iterations:\n")
    (show-Q-table @qt)
    (print "\n\nOptimal policy is:\n")
    (print op)))

;;;------------------------------------------------------------------
;;; Example3 : 8x8 Grid World
;;;------------------------------------------------------------------

;;; Domain: 8 x 8 Grid, Start: top-left, Goal: bottom-right.

;;; https://docs.google.com/drawings/d/1hjFtYNT7IXQS2ObO9_6c1iNaV2eSF4boJfTmvzfbw3M/edit

(defn ex3 []
  (let [grid-world (generate-grid-world 8 8 :s-8-8)
        qt (learnQ :s-1-1 :s-8-8 grid-world 2000)
        op (optimal-policy :s-1-1 :s-8-8 qt)]
    (print "\nLearning Q values for:\n")
    (show-Q-table grid-world)
    (print "\n\nResulting Q values after 2000 iterations:\n")
    (show-Q-table @qt)
    (print "\n\nOptimal policy is:\n")
    (print op)))

;;;------------------------------------------------------------------
;;; Example 4: 4x4 Maze World 
;;;------------------------------------------------------------------

;;; https://docs.google.com/drawings/d/1X6CWfZD7qz0FIBGY055t-wwfqRK3hpBH4gUiAW0XU4g/edit

(def grid-maze {:s-1-1   {:east [:s-1-2 0]}
                :s-1-2   {:east [:s-1-3 0], :west [:s-1-1 0]}
                :s-1-3   {:east [:s-1-4 0], :west [:s-1-2 0]}
                :s-1-4   {:west [:s-1-3 0], :south [:s-2-4 0]}
                :s-2-1   {:east [:s-2-2 0], :south [:s-3-1 0]}
                :s-2-2   {:east [:s-2-3 0], :west [:s-2-1 0]}
                :s-2-3   {:east [:s-2-4 0], :west [:s-2-2 0]}
                :s-2-4   {:west [:s-2-3 0], :north [:s-1-4 0]}
                :s-3-1   {:east [:s-3-2 0], :north [:s-2-1 0], :south [:s-4-1 0]}
                :s-3-2   {:west [:s-3-1 0], :south [:s-4-2 0]}
                :s-3-3   {:east [:s-3-4 0], :south [:s-4-3 0]}
                :s-3-4   {:west [:s-3-3 0], :south [:s-4-4 100]}
                :s-4-1   {:east [:s-4-2 0], :north [:s-3-1 0]}
                :s-4-2   {:east [:s-4-3 0], :west [:s-4-1 0], :north [:s-3-2 0]}
                :s-4-3   {:west [:s-4-2 0], :north [:s-3-3 0]}
                :s-4-4   {:north [:s-3-4 0]}})

;;;------------------------------------------------------------------
  
(defn ex4 []
  (let [qt (learnQ :s-1-1 :s-4-4 grid-maze 2000)
        op (optimal-policy :s-1-1 :s-4-4 qt)]
    (print "\nLearning Q values for:\n")
    (show-Q-table grid-maze)
    (print "\n\nResulting Q values after 2000 iterations:\n")
    (show-Q-table @qt)
    (print "\n\nOptimal policy is:\n")
    (print op)))

;;;------------------------------------------------------------------
;;; End of File
;;;------------------------------------------------------------------
