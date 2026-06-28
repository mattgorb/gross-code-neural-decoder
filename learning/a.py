p = 0.1
s0, s1 = 1, 1                  # the syndrome from the two detectors

bucket = [0.0, 0.0]            # total probability of class 0 and class 1
for a in (0, 1):              # did qubit 0 flip?
    for b in (0, 1):          # did qubit 1 flip?
        for c in (0, 1):      # did qubit 2 flip?
            if (a ^ b) == s0 and (b ^ c) == s1:        # matches both detectors?
                P = (p if a else 1-p) * (p if b else 1-p) * (p if c else 1-p)
                bucket[a ^ b ^ c] += P                  # add to its class

print("coset probs:", bucket, " -> ML class", bucket.index(max(bucket)))






