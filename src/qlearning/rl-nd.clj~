(ns drl.rlnd
  (:require [drl.utils :refer [roundn]]
            [drl.rl :refer [roundn]]))

;;;******************************************************************
;;; Part 1: Non-Deterministic Q-Learning
;;;******************************************************************

;;; This implements the Q-Learning algorithm of Chapter 13 of Tom
;;; Mitchell's book "Machine Learning".

;;;------------------------------------------------------------------

(def gamma 0.9)

;;; Q-Table is a map keyed by state. Each state entry is a map keyed by
;;; action with value next state and reward.

(defn states [table](keys table))
(defn actions [state table](keys (state table)))

;;; The resulting state reached by performing <action> in <state>

(defn next-state [state action table]
  (first (-> table state action)))

(defn immediate-reward [table state action]
  (second (-> table state action)))

;;; This implements the look-ahead of 1 used in Q-Learning in contrast
;;; to the look-ahead of n used in TD-Learning.

(defn state-rewards [Q-table state action]
  (map second (vals ((next-state state action @Q-table) @Q-table))))

;;; This implements argmax of Q-value estimates.

(defn maxQ [Q-table state action]
  (apply max (state-rewards Q-table state action)))

;;;------------------------------------------------------------------
;;; Updaing the Q-Table
;;;------------------------------------------------------------------

;;; This implements the Q-Learning update rule: The immediate reward
;;; plus discounted max Q-value estimate.

(defn update-Q-table [Q-table reward state action]
  (let [new-reward (roundn (+ reward (* gamma (maxQ Q-table state action))))]
    (update-in @Q-table [state action] (fn [val][(first val) new-reward]))))

(defn updateQ! [Q-table reward state action]
  (swap! Q-table (fn [r nw] nw) (update-Q-table Q-table reward state action)))

;;;------------------------------------------------------------------
;;; Random actions and states
;;;------------------------------------------------------------------

(defn random-action [state table]
  (let [actions (actions state table)]
    (nth actions (int (rand (count actions))))))

(defn random-state [table]
  (let [states (states table)]
    (nth states (int (rand (count states))))))

;;;------------------------------------------------------------------
;;; Simple Q-Learning
;;;------------------------------------------------------------------

;;; Basic deterministic Q-learning algorithm using episodes

(defn learnQ [initial-state final-state initial-table iterations]
  (let [Q-table (atom initial-table)]
    (loop [state initial-state
           count 0]
      (cond (>= count iterations)
            Q-table
            ;; Episode ended. Start new episode in random state
            (= state final-state)
            (recur (random-state initial-table) count)
            :else
            (let [action (random-action state initial-table)
                  immediate-reward (immediate-reward initial-table state action)]
              ;; Update Q-Value estimates
              (updateQ! Q-table immediate-reward state action)
              ;; Continue episode with next state
              (recur (next-state state action initial-table)
                     (inc count)))))))

;;;------------------------------------------------------------------
;;; Extracting the Optimal Policy
;;;------------------------------------------------------------------

;;; Retuens a vector of <optimal-action> and <next-state>

(defn optimal-action [state Q-table]
  (let [best (first (sort (fn [[a1 [state-1 val-1]][a2 [staten-2 val-2]]]
                            (> val-1 val-2))
                          (state @Q-table)))]
    [(first best)(first (second best))]))

;;;------------------------------------------------------------------

;;; Returns a vector of alternating <state> and <action>.

(defn optimal-policy [start-state end-state Q-table]
  (loop [result [start-state]
         state start-state
         path-size 0]
    (cond (or (= state end-state)
              (> path-size (count @Q-table)))
          result
          :else
          (let [next-action (optimal-action state Q-table)]
            (recur (vec (concat result next-action))
                   (last next-action)
                   (inc path-size))))))

;;;------------------------------------------------------------------
;;; Output
;;;------------------------------------------------------------------

(defn show-Q-table [table]
  (let [table (into (sorted-map) table)]
    (mapv (fn [state](print "\n" state " " (state table)))
          (keys table))
    nil))

;;;------------------------------------------------------------------
;;; Example 1: Simple Grid from Tom Mitchell's ML book.
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
;;; Examle 2: Arbitrary Grid Worlds
;;;------------------------------------------------------------------

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

(defn generate-grid-world [rows cols fs]
  (let [cells (for [i (range 1 (inc rows)) j (range 1 (inc cols))][i j])]
    (reduce (fn [acc [i j]]
              (assoc acc
                     (make-state-name i j)
                     (make-state-entry i j rows cols fs)))
            {}
            cells)))

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

;;; Domain: 8 x 8 Grid, Start: top-left, Goal: bottom-right.

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
;;; End of File
;;;------------------------------------------------------------------
