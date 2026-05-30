// 02_cache_hot.js — same transaction over and over.
// Only the first request misses the cache; everything else is a hit.
// This measures the cache hit fast path.
//
// Run: k6 run 02_cache_hot.js

import http from 'k6/http'
import { check } from 'k6'

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '5s',  target: 50 },
        { duration: '30s', target: 200 },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '5s',
    },
  },
  thresholds: {
    'http_req_failed': ['rate<0.01'],
  },
}

const features = [
  57007.0, -1.271244, 2.462675, -2.851395, 2.324480, -1.372245,
  -0.948196, -3.065234, 1.166927, -2.268771, -4.881143, 2.255147,
  -4.686387, 0.652375, -6.174288, 0.594380, -4.849692, -6.536521,
  -3.119094, 1.715494, 0.560478, 0.652941, 0.081931, -0.221348,
  -0.523582, 0.224228, 0.756335, 0.632800, 0.250187, 0.010000,
]
const body = JSON.stringify({ features })

export default function () {
  const res = http.post('http://localhost:8080/predict', body, {
    headers: { 'Content-Type': 'application/json' },
    tags: { scenario: 'cache_hot' },
  })
  check(res, {
    'status 200': (r) => r.status === 200,
  })
}
