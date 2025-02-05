########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND sfml_COMPONENT_NAMES sfml-system sfml-main sfml-window sfml-network sfml-audio sfml-graphics)
list(REMOVE_DUPLICATES sfml_COMPONENT_NAMES)
if(DEFINED sfml_FIND_DEPENDENCY_NAMES)
  list(APPEND sfml_FIND_DEPENDENCY_NAMES opengl_system freetype flac OpenAL Vorbis)
  list(REMOVE_DUPLICATES sfml_FIND_DEPENDENCY_NAMES)
else()
  set(sfml_FIND_DEPENDENCY_NAMES opengl_system freetype flac OpenAL Vorbis)
endif()
set(opengl_system_FIND_MODE "NO_MODULE")
set(freetype_FIND_MODE "NO_MODULE")
set(flac_FIND_MODE "NO_MODULE")
set(OpenAL_FIND_MODE "NO_MODULE")
set(Vorbis_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(sfml_PACKAGE_FOLDER_RELEASE "C:/Users/fing.labcom/.conan2/p/sfml2029fced886b7/p")
set(sfml_BUILD_MODULES_PATHS_RELEASE )


set(sfml_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_RES_DIRS_RELEASE )
set(sfml_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_SHARED_LINK_FLAGS_RELEASE )
set(sfml_EXE_LINK_FLAGS_RELEASE )
set(sfml_OBJECTS_RELEASE )
set(sfml_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_COMPILE_OPTIONS_C_RELEASE )
set(sfml_COMPILE_OPTIONS_CXX_RELEASE )
set(sfml_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_BIN_DIRS_RELEASE )
set(sfml_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_LIBS_RELEASE sfml-graphics-s sfml-audio-s sfml-network-s sfml-window-s sfml-main sfml-system-s)
set(sfml_SYSTEM_LIBS_RELEASE ws2_32 gdi32 winmm)
set(sfml_FRAMEWORK_DIRS_RELEASE )
set(sfml_FRAMEWORKS_RELEASE )
set(sfml_BUILD_DIRS_RELEASE )
set(sfml_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(sfml_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_COMPILE_OPTIONS_C_RELEASE}>")
set(sfml_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_EXE_LINK_FLAGS_RELEASE}>")


set(sfml_COMPONENTS_RELEASE sfml-system sfml-main sfml-window sfml-network sfml-audio sfml-graphics)
########### COMPONENT sfml-graphics VARIABLES ############################################

set(sfml_sfml-graphics_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_sfml-graphics_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_sfml-graphics_BIN_DIRS_RELEASE )
set(sfml_sfml-graphics_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_sfml-graphics_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_sfml-graphics_RES_DIRS_RELEASE )
set(sfml_sfml-graphics_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_sfml-graphics_OBJECTS_RELEASE )
set(sfml_sfml-graphics_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_sfml-graphics_COMPILE_OPTIONS_C_RELEASE "")
set(sfml_sfml-graphics_COMPILE_OPTIONS_CXX_RELEASE "")
set(sfml_sfml-graphics_LIBS_RELEASE sfml-graphics-s)
set(sfml_sfml-graphics_SYSTEM_LIBS_RELEASE )
set(sfml_sfml-graphics_FRAMEWORK_DIRS_RELEASE )
set(sfml_sfml-graphics_FRAMEWORKS_RELEASE )
set(sfml_sfml-graphics_DEPENDENCIES_RELEASE sfml-window Freetype::Freetype)
set(sfml_sfml-graphics_SHARED_LINK_FLAGS_RELEASE )
set(sfml_sfml-graphics_EXE_LINK_FLAGS_RELEASE )
set(sfml_sfml-graphics_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-graphics_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-graphics_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-graphics_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-graphics_EXE_LINK_FLAGS_RELEASE}>
)
set(sfml_sfml-graphics_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-graphics_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-graphics_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT sfml-audio VARIABLES ############################################

set(sfml_sfml-audio_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_sfml-audio_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_sfml-audio_BIN_DIRS_RELEASE )
set(sfml_sfml-audio_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_sfml-audio_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_sfml-audio_RES_DIRS_RELEASE )
set(sfml_sfml-audio_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_sfml-audio_OBJECTS_RELEASE )
set(sfml_sfml-audio_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_sfml-audio_COMPILE_OPTIONS_C_RELEASE "")
set(sfml_sfml-audio_COMPILE_OPTIONS_CXX_RELEASE "")
set(sfml_sfml-audio_LIBS_RELEASE sfml-audio-s)
set(sfml_sfml-audio_SYSTEM_LIBS_RELEASE )
set(sfml_sfml-audio_FRAMEWORK_DIRS_RELEASE )
set(sfml_sfml-audio_FRAMEWORKS_RELEASE )
set(sfml_sfml-audio_DEPENDENCIES_RELEASE sfml-system flac::flac OpenAL::OpenAL vorbis::vorbis)
set(sfml_sfml-audio_SHARED_LINK_FLAGS_RELEASE )
set(sfml_sfml-audio_EXE_LINK_FLAGS_RELEASE )
set(sfml_sfml-audio_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-audio_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-audio_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-audio_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-audio_EXE_LINK_FLAGS_RELEASE}>
)
set(sfml_sfml-audio_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-audio_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-audio_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT sfml-network VARIABLES ############################################

set(sfml_sfml-network_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_sfml-network_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_sfml-network_BIN_DIRS_RELEASE )
set(sfml_sfml-network_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_sfml-network_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_sfml-network_RES_DIRS_RELEASE )
set(sfml_sfml-network_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_sfml-network_OBJECTS_RELEASE )
set(sfml_sfml-network_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_sfml-network_COMPILE_OPTIONS_C_RELEASE "")
set(sfml_sfml-network_COMPILE_OPTIONS_CXX_RELEASE "")
set(sfml_sfml-network_LIBS_RELEASE sfml-network-s)
set(sfml_sfml-network_SYSTEM_LIBS_RELEASE ws2_32)
set(sfml_sfml-network_FRAMEWORK_DIRS_RELEASE )
set(sfml_sfml-network_FRAMEWORKS_RELEASE )
set(sfml_sfml-network_DEPENDENCIES_RELEASE sfml-system)
set(sfml_sfml-network_SHARED_LINK_FLAGS_RELEASE )
set(sfml_sfml-network_EXE_LINK_FLAGS_RELEASE )
set(sfml_sfml-network_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-network_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-network_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-network_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-network_EXE_LINK_FLAGS_RELEASE}>
)
set(sfml_sfml-network_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-network_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-network_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT sfml-window VARIABLES ############################################

set(sfml_sfml-window_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_sfml-window_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_sfml-window_BIN_DIRS_RELEASE )
set(sfml_sfml-window_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_sfml-window_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_sfml-window_RES_DIRS_RELEASE )
set(sfml_sfml-window_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_sfml-window_OBJECTS_RELEASE )
set(sfml_sfml-window_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_sfml-window_COMPILE_OPTIONS_C_RELEASE "")
set(sfml_sfml-window_COMPILE_OPTIONS_CXX_RELEASE "")
set(sfml_sfml-window_LIBS_RELEASE sfml-window-s)
set(sfml_sfml-window_SYSTEM_LIBS_RELEASE gdi32 winmm)
set(sfml_sfml-window_FRAMEWORK_DIRS_RELEASE )
set(sfml_sfml-window_FRAMEWORKS_RELEASE )
set(sfml_sfml-window_DEPENDENCIES_RELEASE sfml-system opengl::opengl)
set(sfml_sfml-window_SHARED_LINK_FLAGS_RELEASE )
set(sfml_sfml-window_EXE_LINK_FLAGS_RELEASE )
set(sfml_sfml-window_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-window_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-window_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-window_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-window_EXE_LINK_FLAGS_RELEASE}>
)
set(sfml_sfml-window_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-window_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-window_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT sfml-main VARIABLES ############################################

set(sfml_sfml-main_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_sfml-main_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_sfml-main_BIN_DIRS_RELEASE )
set(sfml_sfml-main_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_sfml-main_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_sfml-main_RES_DIRS_RELEASE )
set(sfml_sfml-main_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_sfml-main_OBJECTS_RELEASE )
set(sfml_sfml-main_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_sfml-main_COMPILE_OPTIONS_C_RELEASE "")
set(sfml_sfml-main_COMPILE_OPTIONS_CXX_RELEASE "")
set(sfml_sfml-main_LIBS_RELEASE sfml-main)
set(sfml_sfml-main_SYSTEM_LIBS_RELEASE )
set(sfml_sfml-main_FRAMEWORK_DIRS_RELEASE )
set(sfml_sfml-main_FRAMEWORKS_RELEASE )
set(sfml_sfml-main_DEPENDENCIES_RELEASE )
set(sfml_sfml-main_SHARED_LINK_FLAGS_RELEASE )
set(sfml_sfml-main_EXE_LINK_FLAGS_RELEASE )
set(sfml_sfml-main_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-main_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-main_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-main_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-main_EXE_LINK_FLAGS_RELEASE}>
)
set(sfml_sfml-main_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-main_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-main_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT sfml-system VARIABLES ############################################

set(sfml_sfml-system_INCLUDE_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/include")
set(sfml_sfml-system_LIB_DIRS_RELEASE "${sfml_PACKAGE_FOLDER_RELEASE}/lib")
set(sfml_sfml-system_BIN_DIRS_RELEASE )
set(sfml_sfml-system_LIBRARY_TYPE_RELEASE STATIC)
set(sfml_sfml-system_IS_HOST_WINDOWS_RELEASE 1)
set(sfml_sfml-system_RES_DIRS_RELEASE )
set(sfml_sfml-system_DEFINITIONS_RELEASE "-DSFML_STATIC")
set(sfml_sfml-system_OBJECTS_RELEASE )
set(sfml_sfml-system_COMPILE_DEFINITIONS_RELEASE "SFML_STATIC")
set(sfml_sfml-system_COMPILE_OPTIONS_C_RELEASE "")
set(sfml_sfml-system_COMPILE_OPTIONS_CXX_RELEASE "")
set(sfml_sfml-system_LIBS_RELEASE sfml-system-s)
set(sfml_sfml-system_SYSTEM_LIBS_RELEASE winmm)
set(sfml_sfml-system_FRAMEWORK_DIRS_RELEASE )
set(sfml_sfml-system_FRAMEWORKS_RELEASE )
set(sfml_sfml-system_DEPENDENCIES_RELEASE )
set(sfml_sfml-system_SHARED_LINK_FLAGS_RELEASE )
set(sfml_sfml-system_EXE_LINK_FLAGS_RELEASE )
set(sfml_sfml-system_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(sfml_sfml-system_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${sfml_sfml-system_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${sfml_sfml-system_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${sfml_sfml-system_EXE_LINK_FLAGS_RELEASE}>
)
set(sfml_sfml-system_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${sfml_sfml-system_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${sfml_sfml-system_COMPILE_OPTIONS_C_RELEASE}>")