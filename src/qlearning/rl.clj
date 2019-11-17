(ns qlearning.rl
  (:require [qlearning.utils :refer [roundn]]))

;;;******************************************************************
;;; Part 1: Basic Deterministic Q-Learning
;;;******************************************************************

;;; This implements the Q-Learning algorithm of Chapter 13 of Tom
;;; Mitchell's book "Machine Learning".

;;; URL: http://profsite.um.ac.ir/~monsefi/machine-learning/pdf/Machine-Learning-Tom-Mitchell.pdf

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
;;; Updating the Q-Table
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

;;; Returns a vector of <optimal-action> and <next-state>

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
;;; End of File
;;;------------------------------------------------------------------
