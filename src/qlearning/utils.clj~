(ns drl.utils)

;;;-----------------------------------------------------------------------
;;; Utilities
;;;-----------------------------------------------------------------------

(defn roundn
  "Rounds a double to the precision of <preision> or 4 if unspecified"
  ([number](roundn number 4))
  ([number precision]
   (let [factor (Math/pow 10 precision)]
     (/ (Math/round (* number factor)) factor))))

(defn dotp "Dot product" [v1 v2] (reduce + (map * v1 v2)))

(defn zipv "Zip vectors" [v1 v2 & vs] (apply mapv vector v1 v2 vs))

(defn transpose [vectors] (apply mapv vector vectors))

(defn sigmoid [x] (/ 1 (+ 1 (Math/pow Math/E (- x)))))

(defn zerov "Returns n zeros" [n](vec (take n (repeat 0))))

;;;-----------------------------------------------------------------------
;;; End of File
;;;-----------------------------------------------------------------------
