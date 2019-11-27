(ns qlearning.core
  (:require [qlearning.examples :refer [ex1 ex2 ex3 ex4]]))


(defn -main
  "Run some Grid world examples of Q-Learning."
  [example]
  (print "\nRunning example number" 2 "\n")
  (cond (= example "1")
        (ex1)
        (= example "2")
        (ex2)
        (= example "3")
        (ex3)
        (= example "4")
        (ex4)
        :else
        (print "Invalid example number. Please use a number between 1 and 4.\n\n")))
