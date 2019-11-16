(ns qlearning.utils)

;;;-----------------------------------------------------------------------
;;; Utilities
;;;-----------------------------------------------------------------------

(defn roundn
  "Rounds a double to the precision of <preision> or 4 if unspecified"
  ([number](roundn number 4))
  ([number precision]
   (let [factor (Math/pow 10 precision)]
     (/ (Math/round (* number factor)) factor))))

;;;-----------------------------------------------------------------------
;;; End of File
;;;-----------------------------------------------------------------------
