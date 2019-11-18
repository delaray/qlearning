(ns qlearning.examples
  (:require [qlearning.rl :refer [learnQ optimal-policy]]))

;;;******************************************************************
;;; Part 1: Helper Functions for examples
;;;******************************************************************

;;;------------------------------------------------------------------
;;; Automatic Grid World Generation
;;;------------------------------------------------------------------

;;; Creates a state name from i & j.

(defn make-state-name [i j]
  (keyword (str "s-" i "-" j)))

;;;------------------------------------------------------------------

;;; Note: Only works for single digit coordinates.

(defn state-coordinates [state-name]
  [(read-string (subs (name state-name) 2 3))
   (read-string (subs (name state-name) 4 5))])

;;;------------------------------------------------------------------

(defn make-state-entry [i j rows cols fs]
  (let [[fs-i fs-j](state-coordinates fs)]
    (apply merge
           `(~(when (< j cols)
                {:east [(make-state-name i (inc j))
                        (if (and (= i fs-i) (= j (dec fs-j))) 100 0)]})
              ~(when (> j 1)
                 {:west [(make-state-name i (dec j))
                         (if (and (= i fs-i)(= j (inc fs-j))) 100 0)]})
              ~(when (> i 1)
                 {:north [(make-state-name (dec i) j)
                          (if (and (= i (inc fs-i)) (= j fs-j)) 100 0)]})
              ~(when (< i rows)
                 {:south [(make-state-name (inc i) j)
                          (if (and (= i (dec fs-i)) (= j fs-j)) 100 0)]})))))

;;;------------------------------------------------------------------

;;; Returns a grid world map with dimensions rows x columns and with
;;; fs as the final state.

(defn generate-grid-world [rows cols fs]
  (let [cells (for [i (range 1 (inc rows)) j (range 1 (inc cols))][i j])]
    (reduce (fn [acc [i j]]
              (assoc acc
                (make-state-name i j)
                (make-state-entry i j rows cols fs)))
            {}
            cells)))

;;;------------------------------------------------------------------
;;; Use-Friendly Output
;;;------------------------------------------------------------------

;;; Displays the specified Q-Table is a user readable manner.

(defn show-Q-table [table]
  (let [table (into (sorted-map) table)]
    (mapv (fn [state](print "\n" state " " (state table)))
          (keys table))
    nil))


;;;------------------------------------------------------------------
;;; Run Example
;;;------------------------------------------------------------------

(defn run-example [start-state goal-state grid-world iterations]
  (let [qt (learnQ start-state goal-state grid-world iterations)
       op (optimal-policy start-state goal-state qt)]
    (print "\nLearning Q values for:\n")
    (show-Q-table grid-world)
    (print "\n\nResulting Q values after " iterations " iterations:\n")
    (show-Q-table @qt)
    (print "\n\nAn optimal policy is:\n")
    (print op)
    (print "\n\n")))

;;;******************************************************************
;;; Part 2: Q-Learning Grid World Examples
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

(defn ex1 [] (run-example :s1 :s3 simple-grid 100))

;;;------------------------------------------------------------------
;;; Example 2: 4x4 Grid World
;;;------------------------------------------------------------------

;;; Domain: 4 x 4 Grid, Start: top-left, Goal: bottom-right.

;;; https://docs.google.com/drawings/d/1hjFtYNT7IXQS2ObO9_6c1iNaV2eSF4boJfTmvzfbw3M

(defn ex2 []
  (let [grid-world (generate-grid-world 4 4 :s-4-4)]
    (run-example :s-1-1 :s-4-4 grid-world 2000)))

;;;------------------------------------------------------------------
;;; Example3 : 8x8 Grid World
;;;------------------------------------------------------------------

;;; Domain: 8 x 8 Grid, Start: top-left, Goal: bottom-right.

(defn ex3 []
  (let [grid-world (generate-grid-world 8 8 :s-8-8)]
    (run-example :s-1-1 :s-8-8 grid-world 2000)))

;;;------------------------------------------------------------------
;;; Example 4: 4x4 Maze World 
;;;------------------------------------------------------------------

;;; Doamin: 4x4 Grid Maze Example, Start: top-left, Goal: bottom-right.

;;; https://docs.google.com/drawings/d/1X6CWfZD7qz0FIBGY055t-wwfqRK3hpBH4gUiAW0XU4g

(def grid-maze {:s-1-1   {:east [:s-1-2 0]}
                :s-1-2   {:east [:s-1-3 0], :west [:s-1-1 0], :south [:s-2-2 0]}
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
  
(defn ex4 [] (run-example :s-1-1 :s-4-4 grid-maze 2000))

;;;------------------------------------------------------------------
;;; End of File
;;;------------------------------------------------------------------
