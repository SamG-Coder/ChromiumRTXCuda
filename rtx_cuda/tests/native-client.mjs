import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
export function connect(executable = process.env.RTXCUDA_HOST || resolve(import.meta.dirname,'../build-native/Release/rtx_cuda_host.exe')) {
  const child = spawn(executable,['--chromium-native-gpu'],{stdio:['pipe','pipe','pipe'],windowsHide:true});
  let bytes = Buffer.alloc(0), id = 0;
  const pending = new Map();
  child.stdout.on('data', chunk => {
    bytes = Buffer.concat([bytes,chunk]);
    while (bytes.length >= 4 && bytes.length >= bytes.readUInt32LE(0) + 4) {
      const length = bytes.readUInt32LE(0), result = JSON.parse(bytes.subarray(4,4+length));
      bytes = bytes.subarray(4+length);
      const request = pending.get(result.id); pending.delete(result.id);
      if (!request) throw Error('Unmatched native reply');
      clearTimeout(request.timer);
      if (result.ok) request.resolve(result.result); else request.reject(Error(result.error));
    }
  });
  let errorOutput = '';
  child.stderr.on('data', x => { errorOutput += x; });
  child.on('exit', code => { for (const item of pending.values()) { clearTimeout(item.timer); item.reject(Error(`Host exited ${code}: ${errorOutput}`)); } pending.clear(); });
  child.on('error', error => { for (const item of pending.values()) { clearTimeout(item.timer); item.reject(error); } pending.clear(); });
  return {
    request(operation,payload = {}) {
      const current = ++id;
      return new Promise((resolve,reject) => {
        const timer = setTimeout(() => { child.kill(); reject(Error('Native request timeout')); },45000);
        pending.set(current,{resolve,reject,timer});
        const data = Buffer.from(JSON.stringify({protocol:1,id:current,operation,payload}));
        const size = Buffer.alloc(4); size.writeUInt32LE(data.length); child.stdin.write(Buffer.concat([size,data]));
      });
    },
    close() { child.stdin.end(); },
    kill() { child.kill(); }
  };
}
