using Base64

original = "original"
println("original: $(original)")
encoded = base64encode(original)
println("encoded: $(encoded)")
decoded = String(base64decode(encoded))
println("decoded: $(decoded)")
