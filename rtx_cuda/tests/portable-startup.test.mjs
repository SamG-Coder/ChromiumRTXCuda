// Start the extracted browser with normal feature defaults, without Playwright's
// launch flags. A debug port only lets the test observe and close its own process.
import assert from 'node:assert/strict';
import {spawn,execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {dirname,join,resolve} from 'node:path';
import {chromium} from 'playwright';
import {serve} from '../scripts/serve.mjs';

const executablePath=process.env.RTXCUDA_CHROME;
assert.ok(executablePath,'Set RTXCUDA_CHROME to an extracted portable chrome.exe');
const root=dirname(resolve(executablePath));
const results=resolve(import.meta.dirname,'../test-results');
const profile=join(root,`startup-test-profile-${Date.now()}`);
const privateFile=join(profile,'private-sentinel.dll');
await mkdir(profile,{recursive:true});
await mkdir(results,{recursive:true});
await writeFile(privateFile,'This is profile data, not a runtime DLL.\n');
const capabilities=[
  'S-1-15-3-1024-3424233489-972189580-2057154623-747635277-1604371224-316187997-3786583170-1043257646',
  'S-1-15-3-1024-2302894289-466761758-1166120688-1039016420-2430351297-4240214049-4028510897-3317428798',
];
function permissions() {
  const script=`$ErrorActionPreference='Stop'
  $paths=@($env:RTXCUDA_ACL_ROOT,(Join-Path $env:RTXCUDA_ACL_ROOT 'chrome.exe'),
    (Join-Path $env:RTXCUDA_ACL_ROOT 'chrome.dll'),(Join-Path $env:RTXCUDA_ACL_ROOT 'locales\\en-US.pak'),
    $env:RTXCUDA_ACL_PROFILE,(Join-Path $env:RTXCUDA_ACL_PROFILE 'private-sentinel.dll'))
  $result=foreach($path in $paths) {
    $acl=Get-Acl -LiteralPath $path
    [pscustomobject]@{path=$path;sddl=$acl.Sddl;aces=@($acl.Access | ForEach-Object {
      [pscustomobject]@{sid=$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value;
        rights=[int]$_.FileSystemRights;inheritance=[int]$_.InheritanceFlags;type=$_.AccessControlType.ToString()}
    })}
  }
  ConvertTo-Json -InputObject @($result) -Depth 5`;
  const env={...process.env,RTXCUDA_ACL_ROOT:root,RTXCUDA_ACL_PROFILE:profile};
  // A parent PowerShell 7 session can otherwise make Windows PowerShell 5 load
  // incompatible modules from the inherited module search path.
  for(const key of Object.keys(env)) if(key.toLowerCase()==='psmodulepath') delete env[key];
  return JSON.parse(execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',script],{
    encoding:'utf8',windowsHide:true,
    env,
  }));
}
const before=permissions();
if(process.env.RTXCUDA_REQUIRE_FRESH_ACL==='1') {
  assert.equal(before.some(item=>item.aces.some(ace=>capabilities.includes(ace.sid))),false,
    'Use a freshly extracted package without preconfigured install capabilities');
}
const report={executablePath,normalStartup:true,checks:[],before,
  executableSha256:createHash('sha256').update(await readFile(executablePath)).digest('hex')};
const server=await serve(0),origin=`http://127.0.0.1:${server.address().port}`;
const args=[`--user-data-dir=${profile}`,'--remote-debugging-port=0','about:blank'];
report.arguments=args;
let browser,child,stderr='';
try {
  child=spawn(executablePath,args,{cwd:root,stdio:['ignore','ignore','pipe']});
  child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-24000);});
  child.on('error',error=>{stderr+=error.stack;});
  let port;
  const deadline=Date.now()+30000;
  while(Date.now()<deadline) {
    assert.equal(child.exitCode,null,`Browser exited: ${child.exitCode}\n${stderr}`);
    try {port=Number((await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);break;}
    catch {await new Promise(resolve=>setTimeout(resolve,200));}
  }
  assert.ok(port,`Normal browser startup timed out\n${stderr}`);
  browser=await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
  const page=browser.contexts()[0].pages()[0];
  await page.goto(origin);
  await page.waitForFunction(()=>document.getElementById('permission').textContent==='Permission: prompt');
  assert.deepEqual(await page.evaluate(async()=>({cuda:await navigator.cuda.SupportsNativeCuda(),
    rtx:await navigator.rtx.SupportsRTX(),permission:await navigator.cuda.queryPermission()})),
    {cuda:true,rtx:true,permission:'prompt'});
  report.checks.push('normal startup loads the local demo and exposes CUDA/RTX with permission prompt');
  const session=await browser.newBrowserCDPSession();
  const system=await session.send('SystemInfo.getInfo');
  assert.equal(system.gpu.auxAttributes.sandboxed,true);
  report.checks.push('GPU sandbox remains enabled');
  await page.goto('chrome://sandbox');
  await page.waitForFunction(()=>document.body.innerText.includes('Network'));
  report.sandbox=await page.locator('body').innerText();
  assert.match(report.sandbox,/Network Service\tNetwork\t[^\n]*S-1-15-2-/,
    'The normal network service must run inside its AppContainer');
  report.checks.push('network service runs in its normal AppContainer sandbox');
  await page.screenshot({path:join(results,'portable-sandbox.png'),fullPage:true});
  report.after=permissions();
  for(const item of report.after.slice(0,4)) {
    for(const sid of capabilities) {
      const grants=item.aces.filter(ace=>ace.sid===sid && ace.type==='Allow');
      assert.equal(grants.length,1,`Expected one capability ACE on ${item.path}`);
      assert.equal(grants[0].rights,0x1200a9,'Capabilities grant only read and execute');
      assert.equal(grants[0].inheritance,0,'Runtime ACEs must not inherit into profiles');
    }
  }
  for(let i=4;i<6;i++) assert.equal(report.after[i].sddl,before[i].sddl,'Profile permissions changed');
  report.checks.push('runtime capability ACEs are read/execute only and do not inherit');
  report.checks.push('profile directory and private file permissions remain unchanged');
  await new Promise(resolve=>setTimeout(resolve,4000));
  assert.equal(child.exitCode,null,'Browser must remain running after startup');
  report.checks.push('browser remains running after normal service startup');
  await session.send('Browser.close').catch(()=>{});
  report.passed=true;
} catch(error) {
  report.passed=false;report.error=error.stack;report.stderr=stderr;process.exitCode=1;
} finally {
  await browser?.close().catch(()=>{});
  // First-run/background work can keep a normal browser alive after a close
  // request. Terminate only the child process created by this test.
  if(child && child.exitCode===null) child.kill();
  server.closeAllConnections();
  server.close();
  await writeFile(join(results,'portable-startup.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({passed:report.passed,checks:report.checks,error:report.error},null,2));
}
