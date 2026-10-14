rule suspicious_apis {
    meta:
        description = "Detects suspicious Windows APIs commonly used by malware"
        author = "Antigravity"
    strings:
        $api1 = "VirtualAlloc" nocase
        $api2 = "VirtualProtect" nocase
        $api3 = "WriteProcessMemory" nocase
        $api4 = "CreateRemoteThread" nocase
        $api5 = "QueueUserAPC" nocase
        $api6 = "SetThreadContext" nocase
        $api7 = "NtCreateThreadEx" nocase
        $api8 = "LdrLoadDll" nocase
    condition:
        any of them
}

rule anti_debugging {
    meta:
        description = "Detects anti-debugging and anti-analysis strings/APIs"
        author = "Antigravity"
    strings:
        $db1 = "IsDebuggerPresent" nocase
        $db2 = "CheckRemoteDebuggerPresent" nocase
        $db3 = "OutputDebugString" nocase
        $db4 = "NtQueryInformationProcess" nocase
        $db5 = "FindWindow" nocase
        $db6 = "GetTickCount" nocase
        $db7 = "QueryPerformanceCounter" nocase
    condition:
        2 of them
}

rule common_packers {
    meta:
        description = "Detects common packers like UPX or ASPack by section names or signatures"
        author = "Antigravity"
    strings:
        $upx1 = "UPX0"
        $upx2 = "UPX1"
        $upx3 = "UPX2"
        $aspack1 = ".aspack"
        $aspack2 = "aspack"
    condition:
        any of them
}

rule suspicious_network_strings {
    meta:
        description = "Detects network indicators like URLs, IPs or User-Agents"
        author = "Antigravity"
    strings:
        $url = /https?:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?/
        $ip = /[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/
        $agent1 = "Mozilla/" nocase
        $agent2 = "User-Agent" nocase
        $cmd = "cmd.exe" nocase
        $powershell = "powershell.exe" nocase
    condition:
        any of them
}
