(ns drl.sync_test
  (:require [clojure.core.matrix :refer :all]
            [drl.annv :as av]
            [drl.annm :as am]
            [drl.utils :refer [roundn]]
            [clojure.test :refer :all]))

;;; Test congruence of results between annv & annm.

;;;-----------------------------------------------------------------------
;;; From IDENTITY FUNCTION EXAMPLE
;;;-----------------------------------------------------------------------

(def nv-nn (av/gen-net [[8 3][3 8]]))

;;; Ensure same initial weights as  nv-nn

(def v-nn (mapv (fn [[bias weights]] [(array bias)(array weights)]) nv-nn))

;;;-----------------------------------------------------------------------

;;; Training Data Identity function example.

(def nvt [[1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]])
(def vt [(array [1 0 0 0 0 0 0 0]) (array [1 0 0 0 0 0 0 0])])

(def nv-i1 (first nvt))
(def nv-o1 (second nvt))

(def v-i1 (first vt))
(def v-o1 (second vt))

;;;-----------------------------------------------------------------------

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
               (mapv vec (am/network-errors v-nn [v-i1 v-o1])))
         (mapv (fn [[b w]][(roundn b)(roundn w)])
               (av/network-errors nv-nn [nv-i1 nv-o1]))))
  ;; Compare non-vectorized weight updates to vectorized version.

  )

;;;-----------------------------------------------------------------------
;;; End of File
;;;-----------------------------------------------------------------------
