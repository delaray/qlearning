(ns drl.annm-test
  (:require [drl.annm :as am]
            [drl.annv :as av]
            [drl.utils :refer [roundn]]
            [clojure.core.matrix :refer :all]
            [clojure.test :refer :all]))

;;;-----------------------------------------------------------------
;;; Sample Neural Nets
;;;-----------------------------------------------------------------

;;; Manually define a simple net with 3 inputs, 2 hidden units, and
;;; 2 output units to verify computations easily and repeatedly.

(def nv-nn
  [[;; Hidden layer Bias vector
    [0.1, 0.1]
    ;; Hidden layer eight matrix
    [[0.2, -0.5, 0.3] [-0.4, 0.2, -0.1]]]
   [;; Output unit bias vector
    [0.1 0.1]
    ;; Output unit weight matrix
    [[0.5,  -0.5] [-0.1, 0.2]]]])

(def v-nn
  (mapv (fn [[bias weights]] [(array bias)(array weights)]) nv-nn))

;;;-----------------------------------------------------------------
;;; Sample Input and Ouputs
;;;-----------------------------------------------------------------

;; Sample non-vecorized and vecorized input vector.
(def nv-i1 [1 2 3])
(def v-i1 (array nv-i1))

;; Sample non-vecorized and vecorized target vector
(def nv-t1 [2 6])
(def v-t1 (array nv-t1))

(def nvt [nv-i1 nv-t1])

(def vt [v-i1 v-t1])

;;;-----------------------------------------------------------------
;;; Tests
;;;-----------------------------------------------------------------

;;; Compare vectorized ann fns to non-vectorized ann fns.

(deftest annm-computations
  ;; Compare non-vecorized network outputs to vectorized version.
  (is (= (mapv roundn (vec (am/network-outputs v-i1 v-nn)))
         (mapv roundn (av/network-outputs nv-i1 nv-nn))))
  ;; Compare non-vecorized all layers outputs to vectorized version.
  (is (= (mapv (fn [outputs](mapv roundn outputs))
               (mapv vec (am/compute-network v-i1 v-nn)))
         (mapv (fn [outputs] (mapv roundn outputs))
               (av/compute-network nv-i1 nv-nn))))
      ;; Compare non-vecorized network errors to vectorized version.
  (is (= (mapv (fn [[b w]][(roundn b)(roundn w)])
               (mapv vec (am/network-errors v-nn [v-i1 v-t1])))
         (mapv (fn [[b w]][(roundn b)(roundn w)])
               (av/network-errors nv-nn [nv-i1 nv-t1]))))
  ;; Compare non-vectorized weight updates to vectorized version.

         )

;;;-----------------------------------------------------------------
;;; End of File
;;;---------------------------------------------------------------
