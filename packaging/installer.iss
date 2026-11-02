; MangaTVL Companion installer (Inno Setup 6). Built by packaging/build_installer.py after build_exe.py.
; Per-user install: no admin rights, no Python, no Docker.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "MangaTVL Companion"
#define AppExe "MangaTVL-Companion.exe"

[Setup]
AppId={{6C3E2B7A-4F1D-4B8E-9A55-0D7E3C1F2A90}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=MangaTVL
DefaultDirName={localappdata}\Programs\MangaTVL
DefaultGroupName=MangaTVL
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=MangaTVL-Companion-Setup-{#AppVersion}
SetupIconFile=build\pyi\mangatvl.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "autostart"; Description: "Start with Windows (เปิดพร้อม Windows)"
Name: "desktopicon"; Description: "Desktop shortcut (ไอคอนบนเดสก์ท็อป)"; Flags: unchecked

[Files]
Source: "dist\MangaTVL-Companion\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; same value as the tray menu's "เปิดพร้อม Windows" (apps/companion/tray.py RUN_NAME)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExe}"""; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start MangaTVL Companion now (เปิดเลย)"; Flags: nowait postinstall

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/im {#AppExe} /f"; Flags: runhidden; RunOnceId: "StopCompanion"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
